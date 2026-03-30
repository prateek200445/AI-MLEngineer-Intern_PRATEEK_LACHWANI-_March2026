from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import re
from config import settings

load_dotenv()
# Import functions from app.py
from app import (
    search_docs,
    build_prompt,
    ask_llm,
    extract_prereq_decision,
    build_answer_from_prereq_decision,
    apply_user_context_to_prereq_decision,
    build_clarification_payload,
    is_eligibility_query,
    extract_user_context_from_text,
    merge_user_context,
)
from eligibility_engine import (
    build_term_plan,
    build_rule_profile,
    evaluate_profile,
    extract_target_course,
    get_profile_from_rule_store,
    load_rule_store,
)
from session_store import PendingClarification, SESSION_STORE

COURSE_CODE_REGEX = r"[A-Z]{1,2}\d{2,3}[A-Z]?"

api = FastAPI(
    title="RAG Search API",
    version="1.0",
    description="API for searching Qdrant + generating answers using Gemini"
)


api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



class QueryRequest(BaseModel):
    question: str
    user_context: dict | None = None
    follow_up_text: str | None = None
    session_id: str | None = None


class TermPlanRequest(BaseModel):
    completed_courses: list[str] | None = None
    follow_up_text: str | None = None
    target_program: str | None = None
    catalog_year: str | None = None
    transfer_credits: int | None = None
    start_term: str = "fall"
    max_courses_per_term: int = 3
    max_credits: int = 12
    term_count: int = 4


def _search_docs_k(query: str, k: int):
    docs = search_docs(query)
    return docs[:k]


def _format_doc_citations(docs):
    citations = []
    for doc, score in docs[:5]:
        metadata = getattr(doc, "metadata", {}) or {}
        source = metadata.get("source") or "unknown-source"
        page = metadata.get("page_label") or metadata.get("page")
        chunk_id = metadata.get("_id")
        heading = metadata.get("title") or (getattr(doc, "page_content", "") or "").split("\n")[0][:80]
        citation_url = f"catalog://{source}" + (f"#page={page}" if page is not None else "")
        citations.append(
            {
                "source": source,
                "page": page,
                "url": citation_url,
                "section_heading": heading,
                "chunk_id": chunk_id,
                "score": float(score),
                "quote": (getattr(doc, "page_content", "") or "")[:220],
            }
        )
    return citations


def _build_abstention_payload(question: str, effective_context: dict, prereq_decision, advanced_evaluation, docs, message: str, missing_field: str):
    return {
        "question": question,
        "answer": "I don't have that information in the provided catalog/policies.",
        "decision": "abstain",
        "reason": message,
        "next_step": "Please check with your advisor, the department page, or the official schedule of classes.",
        "effective_user_context": effective_context,
        "prerequisite_extractor": prereq_decision,
        "advanced_evaluation": advanced_evaluation,
        "clarification": {
            "needs_clarification": True,
            "message": message,
            "missing_fields": [missing_field],
        },
        "documents_used": [
            {
                "score": float(score),
                "content": doc.page_content,
                "metadata": doc.metadata,
            }
            for doc, score in docs
        ],
        "citations": _format_doc_citations(docs),
    }


def _build_next_step(decision: str, target_course: str | None, missing_requirements: list[str] | None = None):
    missing_requirements = missing_requirements or []
    if decision == "eligible":
        return f"You can proceed to enroll in {target_course}."
    if decision == "not_eligible":
        if missing_requirements:
            return "Complete the following first: " + "; ".join(missing_requirements)
        return "Complete at least one required prerequisite before enrolling."
    return "Share missing academic details (completed courses/grades/catalog year) to continue."


def _build_extractive_fallback_answer(question: str, docs, llm_error: Exception | None = None) -> str:
    if not docs:
        return "I could not generate an answer right now, and no supporting catalog context was found. Please try again."

    snippets = []
    for doc, _score in docs[:2]:
        text = (getattr(doc, "page_content", "") or "").strip().replace("\n", " ")
        if text:
            snippets.append(text[:220])

    reason = "I could not use the language model right now"
    if llm_error:
        err = str(llm_error)
        if "429" in err:
            reason = "The language model is temporarily rate-limited"

    lines = [
        f"{reason}. Here is the closest catalog evidence I found:",
    ]
    for idx, snippet in enumerate(snippets, start=1):
        lines.append(f"{idx}. {snippet}")
    lines.append("Please retry in a moment for a fuller synthesized answer.")
    return "\n".join(lines)


def _ask_llm_or_fallback(prompt: str, question: str, docs):
    try:
        return ask_llm(prompt)
    except Exception as exc:
        return _build_extractive_fallback_answer(question, docs, exc)


def _should_resume_pending_intent(question: str, has_pending: bool) -> bool:
    if not has_pending:
        return False
    # If user asks a fresh eligibility question, start a new intent.
    return not is_eligibility_query(question)


def _with_session(payload: dict, session_id: str):
    payload["session_id"] = session_id
    return payload


def _parse_pending_follow_up_reply(question: str, pending: PendingClarification | None) -> dict:
    if not pending:
        return {}

    missing = set(pending.missing_fields or [])
    q = (question or "").strip().lower()
    if not q:
        return {}
    starts_with_no = re.match(r"^\s*no\b", q) is not None

    if "completed_courses" in missing:
        if starts_with_no or q in {"no", "none", "nope", "nah", "n"} or any(
            phrase in q
            for phrase in [
                "no courses",
                "no course codes",
                "none completed",
                "not completed any",
                "have not completed",
                "haven't completed",
            ]
        ):
            return {
                "completed_courses_declared": True,
                "completed_courses": [],
            }

    if "currently_enrolled_courses" not in missing:
        return {}

    if starts_with_no or q in {"no", "nope", "nah", "n"} or any(
        phrase in q
        for phrase in [
            "not currently enrolled",
            "not enrolled",
            "not taking",
            "no co-requisite",
            "no corequisite",
        ]
    ):
        return {
            "currently_enrolled_declared": True,
            "currently_enrolled_courses": [],
        }

    return {}


def _clarification_question_for_field(field: str) -> str:
    prompts = {
        "completed_courses": "Which courses have you already completed? Please provide course codes like AH110, AH152.",
        "additional_completed_courses": "Have you completed any additional required math/CS/core courses? Please share course codes.",
        "currently_enrolled_courses": "Are you currently enrolled in any co-requisite courses? Please list course codes.",
        "grades": "Please share your grades for the prerequisite courses you completed.",
        "gpa": "What is your current GPA?",
        "semester": "Which semester are you in right now? (fall/spring/summer/winter)",
    }
    return prompts.get(field, f"Please provide: {field}.")


def _merge_missing_inputs_into_clarification(clarification: dict | None, missing_inputs: list[str] | None):
    missing_inputs = [m for m in (missing_inputs or []) if m]
    if not missing_inputs:
        return clarification

    if not clarification:
        clarification = {
            "needs_clarification": True,
            "message": "I need a few details before I can give a definitive eligibility answer.",
            "missing_fields": [],
            "follow_up_questions": [],
            "example_context": {
                "completed_courses": ["AH110"],
                "semester": "fall",
                "gpa": 3.4,
            },
        }

    clarification["missing_fields"] = sorted(
        set(clarification.get("missing_fields") or []) | set(missing_inputs)
    )

    follow_ups = clarification.get("follow_up_questions") or []
    existing = {f.get("field") for f in follow_ups if isinstance(f, dict)}
    for field in missing_inputs:
        if field not in existing:
            follow_ups.append({"field": field, "question": _clarification_question_for_field(field)})
    clarification["follow_up_questions"] = follow_ups

    return clarification


def _build_clarification_answer_text(clarification: dict) -> str:
    message = clarification.get("message") or "Need more information before final eligibility decision."
    questions = clarification.get("follow_up_questions") or []
    if not questions:
        return message

    lines = [message, "", "Please reply with:"]
    for idx, item in enumerate(questions, start=1):
        q = (item or {}).get("question")
        if q:
            lines.append(f"{idx}. {q}")
    return "\n".join(lines)


def _should_parse_context_from_question(question: str) -> bool:
    q = (question or "").lower()
    has_context_markers = any(
        marker in q
        for marker in [
            "completed",
            "completing",
            "passed",
            "finished",
            "done",
            "enrolled",
            "taking",
            "instructor consent",
            "consent of instructor",
            "gpa",
            "cgpa",
            "grade point average",
            "grade",
            "semester",
            "fall",
            "spring",
            "summer",
            "winter",
        ]
    )
    if has_context_markers:
        return True

    # Allow short, non-question course-code messages as context replies, e.g. "AH110, AH152".
    has_course_code = re.search(rf"\b{COURSE_CODE_REGEX}\b", (question or "").upper()) is not None
    looks_like_question = any(
        marker in q
        for marker in [
            "can i",
            "am i",
            "eligible",
            "prerequisite",
            "what",
            "which",
            "how",
            "take",
            "enroll",
            "register",
            "?",
        ]
    )
    return has_course_code and not looks_like_question


def _is_prereq_or_eligibility_intent(question: str) -> bool:
    q = (question or "").lower()
    if is_eligibility_query(question) or "prerequisite" in q:
        return True
    return any(
        marker in q
        for marker in [
            "what courses do i need before",
            "what do i need before",
            "full prerequisite path",
            "prerequisite chain",
            "directly take",
            "before taking",
            "override prerequisites",
        ]
    )


def _is_course_recommendation_query(question: str) -> bool:
    q = (question or "").lower()
    return any(
        marker in q
        for marker in [
            "what courses can i take after",
            "what can i take after",
            "courses can i take after",
            "after completing",
            "after i complete",
        ]
    )


def _build_course_recommendation_answer(completed_courses: list[str], suggested_courses: list[str], start_term: str) -> str:
    if not suggested_courses:
        return (
            "Based on the current rule set, I could not find unlocked next courses right now. "
            "Please verify your completed courses or ask for a specific target course path."
        )

    completed_text = ", ".join(completed_courses)
    suggested_text = ", ".join(suggested_courses)
    return (
        f"Based on your completed courses ({completed_text}), the next courses you can consider in {start_term} are: "
        f"{suggested_text}."
    )


def _is_non_catalog_fact_query(question: str) -> bool:
    q = (question or "").lower()
    return any(
        marker in q
        for marker in [
            "who teaches",
            "instructor",
            "difficulty of",
            "admission gpa",
            "gpa required for",
        ]
    )


def _explicit_negative_prereq_signal(question: str) -> tuple[bool, list[str]]:
    q = (question or "")
    q_lower = q.lower()
    negated_codes = []
    negated_sections = re.findall(r"(?:not\s+completed|without\s+completing|haven'?t\s+completed)\s+([^?.!]+)", q_lower)
    for section in negated_sections:
        negated_codes.extend(re.findall(r"\b[A-Z]{2}\d{2,3}\b", section.upper()))
    negated_codes = sorted(set(negated_codes))

    has_explicit_required_phrase = "not completed required prerequisites" in q_lower
    # Conservative rule:
    # - explicit "required prerequisites" negative => not eligible
    # - negating 2+ specific prerequisite options => not eligible
    signal = has_explicit_required_phrase or len(negated_codes) >= 2
    return signal, negated_codes


def _has_min_evidence(question: str, docs) -> bool:
    return _has_min_evidence_with_threshold(question, docs, min_overlap=1)


def _has_min_evidence_with_threshold(question: str, docs, min_overlap: int) -> bool:
    if not docs:
        return False
    q_tokens = {t for t in re.findall(r"[a-z0-9]+", (question or "").lower()) if len(t) > 3}
    # Ignore generic query words.
    q_tokens = {t for t in q_tokens if t not in {"what", "which", "where", "when", "policy", "course", "eligible", "prerequisite", "credits"}}
    if not q_tokens:
        return True

    merged_text = " ".join([(getattr(doc, "page_content", "") or "") for doc, _score in docs[:5]]).lower()
    overlap = sum(1 for t in q_tokens if t in merged_text)
    return overlap >= max(1, int(min_overlap))


def _policy_thresholds(question: str):
    is_elig = is_eligibility_query(question) or ("prerequisite" in (question or "").lower())
    if is_elig:
        return {
            "intent": "eligibility",
            "confidence_threshold": float(settings.ELIGIBILITY_RAG_CONFIDENCE_THRESHOLD),
            "min_overlap": int(settings.ELIGIBILITY_MIN_TOKEN_OVERLAP),
        }
    return {
        "intent": "policy",
        "confidence_threshold": float(settings.POLICY_RAG_CONFIDENCE_THRESHOLD),
        "min_overlap": int(settings.POLICY_MIN_TOKEN_OVERLAP),
    }


RULE_STORE_PATH = os.path.join(os.path.dirname(__file__), "catalog_rules.json")
RULE_STORE = {}
RULE_STORE_MTIME = None


def _get_rule_store():
    global RULE_STORE, RULE_STORE_MTIME

    try:
        mtime = os.path.getmtime(RULE_STORE_PATH)
    except Exception:
        mtime = None

    if RULE_STORE_MTIME != mtime:
        RULE_STORE = load_rule_store(RULE_STORE_PATH)
        RULE_STORE_MTIME = mtime

    return RULE_STORE


def _needs_catalog_year() -> bool:
    """Catalog year is only required when multiple catalog PDFs are present."""
    try:
        policy_dir = os.path.join(os.path.dirname(__file__), settings.PDF_FOLDER)
        pdfs = [f for f in os.listdir(policy_dir) if f.lower().endswith(".pdf")]
        return len(pdfs) > 1
    except Exception:
        # Be conservative if folder lookup fails.
        return True


@api.get("/")
def home():
    return {"status": "RAG API is running!"}


@api.post("/query")
def query_rag(req: QueryRequest):
    try:
        session = SESSION_STORE.get_or_create(req.session_id)

        incoming_question = req.question
        resume_pending = _should_resume_pending_intent(
            incoming_question,
            session.pending_clarification is not None,
        )
        evaluation_question = (
            session.pending_clarification.original_question
            if resume_pending and session.pending_clarification
            else incoming_question
        )

        parsed_follow_up_context = extract_user_context_from_text(req.follow_up_text)
        parsed_question_context = (
            extract_user_context_from_text(incoming_question)
            if _should_parse_context_from_question(incoming_question)
            else {}
        )
        pending_reply_context = _parse_pending_follow_up_reply(
            incoming_question,
            session.pending_clarification,
        )
        effective_context = merge_user_context(session.current_user_context, req.user_context)
        effective_context = merge_user_context(effective_context, parsed_follow_up_context)
        effective_context = merge_user_context(effective_context, parsed_question_context)
        effective_context = merge_user_context(effective_context, pending_reply_context)
        session.current_user_context = effective_context
        session.touch()

        question = evaluation_question

        # Special handling: recommendation intent should return actionable next-course suggestions,
        # not strict eligibility clarification loops.
        if _is_course_recommendation_query(question):
            rule_store = _get_rule_store()
            completed_courses = sorted(
                {
                    c.upper()
                    for c in (effective_context.get("completed_courses") or [])
                    if isinstance(c, str) and re.match(rf"^{COURSE_CODE_REGEX}$", c.upper())
                }
            )

            if not completed_courses:
                clarification = _merge_missing_inputs_into_clarification(None, ["completed_courses"])
                session.pending_clarification = PendingClarification(
                    original_question=question,
                    target_course=None,
                    missing_fields=list(clarification.get("missing_fields") or []),
                    follow_up_questions=list(clarification.get("follow_up_questions") or []),
                    message=clarification.get("message"),
                )
                session.touch()
                return _with_session({
                    "question": question,
                    "answer": _build_clarification_answer_text(clarification),
                    "decision": "need_more_info",
                    "next_step": "Share completed course codes so I can suggest your next unlocked courses.",
                    "effective_user_context": effective_context,
                    "prerequisite_extractor": None,
                    "advanced_evaluation": None,
                    "clarification": clarification,
                    "documents_used": [],
                    "citations": [],
                }, session.session_id)

            if not rule_store:
                raise HTTPException(status_code=500, detail="No catalog rule store is loaded. Run ingest.py to generate catalog_rules.json.")

            start_term = (effective_context.get("semester") or "fall").lower()
            plan = build_term_plan(
                rule_store=rule_store,
                completed_courses=completed_courses,
                start_term=start_term,
                max_courses_per_term=5,
                max_credits=20,
                term_count=1,
            )
            suggested = []
            if plan.get("status") == "ok":
                first_term = (plan.get("planned_terms") or [{}])[0]
                suggested = [c.get("course") for c in (first_term.get("courses") or []) if c.get("course")]

            answer = _build_course_recommendation_answer(completed_courses, suggested, start_term)
            session.pending_clarification = None
            session.touch()
            return _with_session({
                "question": question,
                "answer": answer,
                "decision": "policy_answer",
                "next_step": "Ask for any specific course to get an eligibility check.",
                "effective_user_context": effective_context,
                "prerequisite_extractor": None,
                "advanced_evaluation": {
                    "mode": "course_recommendation",
                    "completed_courses": completed_courses,
                    "suggested_courses": suggested,
                },
                "clarification": None,
                "documents_used": [],
                "citations": [],
            }, session.session_id)

        # 1. Search documents
        docs = search_docs(question)

        # 2. Build prompt for LLM
        final_prompt = build_prompt(docs, question)

        # 3. Deterministic prerequisite extraction first, then fallback to LLM
        prereq_decision = extract_prereq_decision(question, docs)
        prereq_decision = apply_user_context_to_prereq_decision(prereq_decision, effective_context)

        target_course = None
        if prereq_decision and prereq_decision.get("target_course"):
            target_course = prereq_decision.get("target_course")
        else:
            target_course = extract_target_course(question)

        advanced_evaluation = None
        course_in_rule_store = False
        if target_course:
            rule_store = _get_rule_store()
            course_in_rule_store = target_course.upper() in rule_store
            rule_profile = get_profile_from_rule_store(target_course, rule_store)
            rule_source = "catalog_rules.json"
            if not rule_profile:
                rule_profile = build_rule_profile(
                    target_course=target_course,
                    docs=docs,
                    search_fn=_search_docs_k,
                    depth=2,
                )
                rule_source = "rag-fallback"

            result = evaluate_profile(rule_profile, effective_context)
            advanced_evaluation = {
                "target_course": target_course,
                "source": rule_source,
                "rule_profile": {
                    "any_of": rule_profile.any_of,
                    "all_of": rule_profile.all_of,
                    "co_requisites": rule_profile.co_requisites,
                    "min_grade_by_course": rule_profile.min_grade_by_course,
                    "allows_instructor_consent": rule_profile.allows_instructor_consent,
                    "prerequisite_chain": rule_profile.prerequisite_chain,
                    "confidence": rule_profile.confidence,
                    "evidence": rule_profile.evidence[:5],
                },
                "result": {
                    "decision": result.decision,
                    "reasons": result.reasons,
                    "missing_requirements": result.missing_requirements,
                    "missing_inputs": result.missing_inputs,
                },
            }

        prereq_or_eligibility_intent = _is_prereq_or_eligibility_intent(question)
        has_final_deterministic_decision = (
            (prereq_decision is not None and prereq_decision.get("eligibility") in [True, False])
            or (
                advanced_evaluation is not None
                and advanced_evaluation.get("result", {}).get("decision") in ["eligible", "not_eligible"]
            )
        )

        clarification = build_clarification_payload(prereq_decision, question, effective_context)
        if advanced_evaluation and advanced_evaluation.get("result", {}).get("missing_inputs"):
            clarification = _merge_missing_inputs_into_clarification(
                clarification,
                advanced_evaluation["result"].get("missing_inputs") or [],
            )

        neg_signal, negated_codes = _explicit_negative_prereq_signal(question)
        if prereq_or_eligibility_intent and neg_signal:
            missing_reqs = []
            if negated_codes:
                missing_reqs = ["Complete at least one of: " + ", ".join(negated_codes)]
            answer = (
                "Final answer: Not eligible\n"
                "Why: You explicitly indicated missing prerequisite completion in your question."
            )
            if missing_reqs:
                answer += "\nMissing requirements: " + "; ".join(missing_reqs)

            return _with_session({
                "question": question,
                "answer": answer,
                "decision": "not_eligible",
                "next_step": _build_next_step("not_eligible", target_course, missing_reqs),
                "effective_user_context": effective_context,
                "prerequisite_extractor": prereq_decision,
                "advanced_evaluation": advanced_evaluation,
                "clarification": None,
                "documents_used": [
                    {
                        "score": float(score),
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                    }
                    for doc, score in docs
                ],
                "citations": _format_doc_citations(docs),
                "thresholds_used": _policy_thresholds(question),
            }, session.session_id)

        if prereq_or_eligibility_intent and not clarification:
            profile_confidence = 0.0
            evidence_count = 0
            if advanced_evaluation:
                profile_confidence = float(advanced_evaluation.get("rule_profile", {}).get("confidence", 0.0) or 0.0)
                evidence_count = len(advanced_evaluation.get("rule_profile", {}).get("evidence") or [])

            # Conservative guardrail: do not finalize prereq outcomes on weak/partial evidence.
            if (not has_final_deterministic_decision) or profile_confidence < 0.75 or evidence_count < 2:
                ask_field = "additional_completed_courses" if (effective_context.get("completed_courses") or []) else "completed_courses"
                clarification = _merge_missing_inputs_into_clarification(None, [ask_field])

        if clarification:
            session.pending_clarification = PendingClarification(
                original_question=question,
                target_course=target_course,
                missing_fields=list(clarification.get("missing_fields") or []),
                follow_up_questions=list(clarification.get("follow_up_questions") or []),
                message=clarification.get("message"),
            )
            session.touch()

            # Ask follow-up questions instead of giving a potentially unreliable final eligibility answer.
            return _with_session({
                "question": question,
                "answer": _build_clarification_answer_text(clarification),
                "decision": "need_more_info",
                "next_step": _build_next_step("need_more_info", target_course),
                "effective_user_context": effective_context,
                "prerequisite_extractor": prereq_decision,
                "advanced_evaluation": advanced_evaluation,
                "clarification": clarification,
                "documents_used": [
                    {
                        "score": float(score),
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                    }
                    for doc, score in docs
                ],
                "citations": _format_doc_citations(docs),
            }, session.session_id)

        if session.pending_clarification:
            session.pending_clarification = None
            session.touch()

        answer = build_answer_from_prereq_decision(prereq_decision) if prereq_or_eligibility_intent else None
        prereq_has_explicit_decision = (
            prereq_decision is not None and prereq_decision.get("eligibility") in [True, False]
        )
        if (
            advanced_evaluation
            and advanced_evaluation.get("result", {}).get("decision") == "eligible"
            and prereq_or_eligibility_intent
            and not prereq_has_explicit_decision
        ):
            reqs = advanced_evaluation.get("rule_profile", {}).get("any_of") or []
            answer = (
                "Final answer: Eligible\n"
                "Why: Deterministic rule evaluation found all detected constraints satisfied."
                + (f" One-of prerequisite options: {', '.join(reqs)}." if reqs else "")
            )

        if (
            advanced_evaluation
            and advanced_evaluation.get("result", {}).get("decision") == "not_eligible"
            and prereq_or_eligibility_intent
            and not prereq_has_explicit_decision
        ):
            missing_reqs = advanced_evaluation.get("result", {}).get("missing_requirements") or []
            answer = (
                "Final answer: Not eligible\n"
                "Why: Deterministic rule evaluation found unmet requirements.\n"
                "Missing requirements: " + "; ".join(missing_reqs)
            )

        # Strict no-guess policy for prerequisite/eligibility queries.
        thresholds = _policy_thresholds(question)
        if _is_non_catalog_fact_query(question):
            payload = _build_abstention_payload(
                question=question,
                effective_context=effective_context,
                prereq_decision=prereq_decision,
                advanced_evaluation=advanced_evaluation,
                docs=docs,
                message="This question requires information (instructor assignment, comparative difficulty, or admission policy) that may not be explicitly stated in catalog excerpts.",
                missing_field="policy_scope",
            )
            payload["thresholds_used"] = thresholds
            return _with_session(payload, session.session_id)

        strict_policy_intent = prereq_or_eligibility_intent
        low_confidence = (
            advanced_evaluation
            and advanced_evaluation.get("source") == "rag-fallback"
            and float(advanced_evaluation.get("rule_profile", {}).get("confidence", 0.0) or 0.0)
            < float(thresholds["confidence_threshold"])
        )
        weak_evidence = not (advanced_evaluation and advanced_evaluation.get("rule_profile", {}).get("evidence"))
        no_deterministic_answer = not answer
        if strict_policy_intent and (no_deterministic_answer or low_confidence or weak_evidence):
            clarification = _merge_missing_inputs_into_clarification(None, ["additional_completed_courses"])
            session.pending_clarification = PendingClarification(
                original_question=question,
                target_course=target_course,
                missing_fields=list(clarification.get("missing_fields") or []),
                follow_up_questions=list(clarification.get("follow_up_questions") or []),
                message=clarification.get("message"),
            )
            session.touch()
            return _with_session({
                "question": question,
                "answer": _build_clarification_answer_text(clarification),
                "decision": "need_more_info",
                "next_step": _build_next_step("need_more_info", target_course),
                "effective_user_context": effective_context,
                "prerequisite_extractor": prereq_decision,
                "advanced_evaluation": advanced_evaluation,
                "clarification": clarification,
                "documents_used": [
                    {
                        "score": float(score),
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                    }
                    for doc, score in docs
                ],
                "citations": _format_doc_citations(docs),
                "thresholds_used": thresholds,
            }, session.session_id)

        if not answer:
            if target_course and not course_in_rule_store:
                payload = _build_abstention_payload(
                    question=question,
                    effective_context=effective_context,
                    prereq_decision=prereq_decision,
                    advanced_evaluation=advanced_evaluation,
                    docs=docs,
                    message="Please verify the course code or provide an official policy excerpt.",
                    missing_field="valid_course_code",
                )
                payload["thresholds_used"] = thresholds
                return _with_session(payload, session.session_id)

            # For policy queries, be more lenient: if we have documents, let LLM try to answer
            # Only abstain if there's truly no evidence at all
            intent = thresholds["intent"]
            if intent == "policy" and docs:
                # For general policy queries with relevant docs, attempt LLM answer
                answer = _ask_llm_or_fallback(final_prompt, question, docs)
            elif not _has_min_evidence_with_threshold(question, docs, thresholds["min_overlap"]):
                payload = _build_abstention_payload(
                    question=question,
                    effective_context=effective_context,
                    prereq_decision=prereq_decision,
                    advanced_evaluation=advanced_evaluation,
                    docs=docs,
                    message="No verifiable evidence was found in the retrieved policy context.",
                    missing_field="policy_evidence",
                )
                payload["thresholds_used"] = thresholds
                return _with_session(payload, session.session_id)
            else:
                answer = _ask_llm_or_fallback(final_prompt, question, docs)

        # 4. Format document details
        result_docs = [
            {
                "score": float(score),
                "content": doc.page_content,
                "metadata": doc.metadata
            }
            for doc, score in docs
        ]

        final_decision = None
        missing_reqs = []
        if prereq_or_eligibility_intent and prereq_decision and prereq_decision.get("eligibility") is True:
            final_decision = "eligible"
        elif prereq_or_eligibility_intent and prereq_decision and prereq_decision.get("eligibility") is False:
            final_decision = "not_eligible"
        elif prereq_or_eligibility_intent and advanced_evaluation:
            final_decision = advanced_evaluation.get("result", {}).get("decision")
            missing_reqs = advanced_evaluation.get("result", {}).get("missing_requirements") or []

        return _with_session({
            "question": question,
            "answer": answer,
            "decision": final_decision,
            "next_step": _build_next_step(final_decision or "need_more_info", target_course, missing_reqs),
            "effective_user_context": effective_context,
            "prerequisite_extractor": prereq_decision,
            "advanced_evaluation": advanced_evaluation,
            "clarification": None,
            "thresholds_used": thresholds,
            "documents_used": result_docs,
            "citations": _format_doc_citations(docs),
            "prompt_sent_to_llm": final_prompt
        }, session.session_id)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api.post("/extract-prerequisite")
def extract_prerequisite(req: QueryRequest):
    """Returns only structured prerequisite extraction output."""
    try:
        docs = search_docs(req.question)
        decision = extract_prereq_decision(req.question, docs)
        return {
            "question": req.question,
            "prerequisite_extractor": decision,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api.post("/plan-term")
def plan_term(req: TermPlanRequest):
    """Generate a deterministic multi-term plan from completed courses and catalog rules."""
    try:
        parsed = extract_user_context_from_text(req.follow_up_text)
        extracted_completed = parsed.get("completed_courses", []) if isinstance(parsed, dict) else []
        raw_completed = [c.upper().strip() for c in (req.completed_courses or [])] + [c.upper().strip() for c in extracted_completed]
        completed = sorted(set([c for c in raw_completed if re.match(rf"^{COURSE_CODE_REGEX}$", c)]))

        missing = []
        questions = []
        if not req.target_program:
            missing.append("target_program")
            questions.append("What is your target program/major?")
        if _needs_catalog_year() and not req.catalog_year:
            missing.append("catalog_year")
            questions.append("Which catalog year should I use?")
        if req.transfer_credits is None:
            missing.append("transfer_credits")
            questions.append("How many transfer credits should be considered?")
        # Allow planning from scratch (no completed courses), but ignore invalid tokens.

        rule_store = _get_rule_store()
        if not rule_store:
            return {
                "status": "error",
                "message": "No catalog rule store is loaded. Run ingest.py to generate catalog_rules.json.",
            }

        target_text = (req.target_program or "").strip()
        target = target_text.upper()
        target_hint_course = None
        target_hint_prefix = None

        # Accept target course hints embedded in labels like "CS51: Formal Analyses" or "NS110L".
        code_match = re.search(rf"\b({COURSE_CODE_REGEX})\b", target)
        invalid_target_code = False
        if code_match:
            extracted_code = code_match.group(1).upper()
            prefix_match = re.match(r"^([A-Z]{1,2})", extracted_code)
            prefix = prefix_match.group(1) if prefix_match else None
            target_hint_prefix = prefix
            candidate_codes = [extracted_code]

            # Compatibility fallback: allow CS51-style input to match CS051 if present.
            m_digits = re.match(r"^([A-Z]{1,2})(\d{2})([A-Z]?)$", extracted_code)
            if m_digits:
                candidate_codes.append(f"{m_digits.group(1)}0{m_digits.group(2)}{m_digits.group(3)}")

            matched = next((c for c in candidate_codes if c in rule_store), None)
            if matched:
                target_hint_course = matched
            else:
                invalid_target_code = True
        elif "computer science" in target.lower():
            target_hint_prefix = "CS"

        if invalid_target_code:
            missing.append("target_program")
            questions.append("I could not find that target course code in the catalog rules. Please provide a valid target program/course (e.g., CS110 or 'Computational Sciences').")

        if missing:
            response = {
                "status": "needs_clarification",
                "message": "I need a few details before creating a reliable term plan.",
                "missing_fields": missing[:5],
                "clarifying_questions": questions[:5],
            }
            if invalid_target_code:
                suggested = sorted([k for k in rule_store.keys() if (target_hint_prefix and k.startswith(target_hint_prefix))])
                if not suggested:
                    suggested = sorted(rule_store.keys())
                response["suggested_target_courses"] = suggested[:40]
            return response

        plan = build_term_plan(
            rule_store=rule_store,
            completed_courses=completed,
            target_course=target_hint_course,
            target_prefix=target_hint_prefix,
            start_term=req.start_term,
            max_courses_per_term=req.max_courses_per_term,
            max_credits=req.max_credits,
            term_count=req.term_count,
        )
        if plan.get("status") == "ok":
            first_term = (plan.get("planned_terms") or [{}])[0]
            plan["suggested_next_term_courses"] = [c.get("course") for c in first_term.get("courses", [])]
        effective_catalog_year = req.catalog_year if req.catalog_year else "single-catalog-active"
        plan["input"] = {
            "completed_courses": completed,
            "target_program": req.target_program,
            "catalog_year": effective_catalog_year,
            "transfer_credits": req.transfer_credits,
            "start_term": req.start_term,
            "max_courses_per_term": req.max_courses_per_term,
            "max_credits": req.max_credits,
            "term_count": req.term_count,
        }
        return plan
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
