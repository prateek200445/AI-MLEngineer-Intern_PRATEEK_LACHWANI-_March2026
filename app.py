from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from qdrant_client import QdrantClient
from langchain_google_genai import ChatGoogleGenerativeAI

import subprocess
import re
import json
import time
from urllib import request, error
from config import settings


ELIGIBILITY_PROMPT_TEMPLATE = """
Answer the question using only the context below.

Rules:
1) Do not infer missing facts.
2) If the context does not explicitly state the prerequisite/eligibility for the exact target course, say you cannot confirm.
3) For prerequisite questions, prefer explicit prerequisite statements for that course over nearby mentions.
4) Keep the answer short and direct.

{context}

---

Question: {question}

Return:
- Final answer
- Why (1-2 lines)
- Citations (quote short supporting lines from context)
"""


GENERAL_CATALOG_PROMPT_TEMPLATE = """
Answer the question using only the catalog/policy context below.

Rules:
1) Answer clearly and directly.
2) If the exact information is not present in context, say so briefly.
3) Do not fabricate numbers, dates, policies, or requirements.
4) Keep the answer concise and user-friendly.

{context}

---

Question: {question}

Return:
- Final answer
- Supporting evidence (1-2 short lines)
- Citations (quote short supporting lines from context)
"""


COURSE_CODE_REGEX = r"[A-Z]{1,2}\d{2,3}[A-Z]?"
COURSE_CODE_PATTERN = re.compile(rf"\b{COURSE_CODE_REGEX}\b")


def _extract_course_codes(text: str):
    return list(dict.fromkeys(COURSE_CODE_PATTERN.findall((text or "").upper())))


def _token_overlap(query: str, text: str) -> float:
    q_tokens = {t for t in re.findall(r"[a-z0-9]+", (query or "").lower()) if len(t) > 2}
    d_tokens = {t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(t) > 2}
    if not q_tokens:
        return 0.0
    return len(q_tokens & d_tokens) / len(q_tokens)


def _prereq_bonus(query: str, text: str, target_course: str | None) -> float:
    q = (query or "").lower()
    t = (text or "").lower()
    bonus = 0.0

    prereq_query = any(
        keyword in q
        for keyword in [
            "prerequisite",
            "prerequisites",
            "eligible",
            "can i take",
            "can i enroll",
            "enroll in",
            "can i register",
            "register for",
            "if i have not",
            "without completing",
            "require",
            "required",
        ]
    )

    if not prereq_query:
        return bonus

    if "prerequisite" in t or "prerequisites" in t:
        bonus += 1.5

    if target_course:
        target_lower = target_course.lower()
        if target_lower in t:
            bonus += 1.2

        explicit_patterns = [
            rf"{target_lower}[^\n:.]*prerequisites?",
            rf"prerequisites?[^\n:.]*{target_lower}",
            rf"{target_lower}[^\n:.]*(requires|required)",
        ]
        if any(re.search(pat, t) for pat in explicit_patterns):
            bonus += 2.0

    return bonus

# ------------------ EMBEDDINGS (BGE – KEPT) ------------------
embeddings = HuggingFaceBgeEmbeddings(
    model_name=settings.EMBEDDING_MODEL,
    model_kwargs={"device": settings.EMBEDDING_DEVICE},
    encode_kwargs={"normalize_embeddings": settings.EMBEDDING_NORMALIZE}
)

# ------------------ QDRANT CONNECTION ------------------
url = settings.QDRANT_URL or f"http://{settings.QDRANT_HOST}:{settings.QDRANT_PORT}"

client = QdrantClient(
    url=url,
    prefer_grpc=settings.PREFER_GRPC,
    api_key=settings.QDRANT_API_KEY
)

print("✅ Qdrant client connected")

db = Qdrant(
    client=client,
    collection_name=settings.QDRANT_COLLECTION,
    embeddings=embeddings   # ← KEEP THIS (important for your version)
)

print("✅ Qdrant vector store ready")

# ------------------ SEARCH ------------------
def search_docs(query: str):
    # For policy queries with potential wording mismatches, try multiple query variations
    queries_to_try = [query]
    q_lower = (query or "").lower()
    
    # Add expanded queries for common policy questions
    if "minor" in q_lower and ("max" in q_lower or "how many" in q_lower or "allowed" in q_lower):
        queries_to_try.append("number of minors policy")
        queries_to_try.append("minors limit students declare")
        queries_to_try.append("concurrent minors regulations")
    
    # Try each query and collect results
    all_candidates = []
    for search_query in queries_to_try:
        candidates = db.similarity_search_with_score(search_query, k=15)
        all_candidates.extend(candidates)
    
    if not all_candidates:
        # Fallback to original query
        all_candidates = db.similarity_search_with_score(query, k=25)
    
    if not all_candidates:
        return []

    # Remove duplicates (same document content)
    seen = set()
    unique_candidates = []
    for doc, score in all_candidates:
        content_hash = hash(doc.page_content[:100])  # Use first 100 chars as identifier
        if content_hash not in seen:
            seen.add(content_hash)
            unique_candidates.append((doc, score))

    course_codes = _extract_course_codes(query)
    target_course = course_codes[0] if course_codes else None

    reranked = []
    for rank, (doc, raw_score) in enumerate(unique_candidates[:25]):  # Limit to top 25 before reranking
        text = doc.page_content or ""
        rank_signal = 1.0 / (rank + 1)
        lexical_signal = _token_overlap(query, text)
        prereq_signal = _prereq_bonus(query, text, target_course)

        combined_score = rank_signal + (0.35 * lexical_signal) + prereq_signal
        reranked.append((doc, combined_score, raw_score))

    reranked.sort(key=lambda x: x[1], reverse=True)
    top = reranked[:5]
    return [(doc, score) for doc, score, _raw in top]

# ------------------ PROMPT ------------------
def build_prompt(docs, query):
    if not docs:
        return f"No relevant context found. Answer the question:\n{query}"

    context_text = "\n\n---\n\n".join(
        doc.page_content for doc, _ in docs
    )

    is_eligibility = _is_prereq_intent(query)
    template = ELIGIBILITY_PROMPT_TEMPLATE if is_eligibility else GENERAL_CATALOG_PROMPT_TEMPLATE
    prompt = ChatPromptTemplate.from_template(template)
    return prompt.format(context=context_text, question=query)

# ------------------ LLM ------------------
def get_llm():
    provider = settings.LLM_PROVIDER.lower()

    if provider == "google":
        return ChatGoogleGenerativeAI(
            model=settings.GENERATIVE_MODEL_NAME,
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=0.2,
            max_output_tokens=1024
        )

    elif provider == "openai":
        from langchain.chat_models import ChatOpenAI
        return ChatOpenAI(
            model=settings.OPENAI_MODEL_NAME,
            api_key=settings.OPENAI_API_KEY
        )

    elif provider == "openrouter":
        # OpenRouter is called directly in ask_llm to avoid additional SDK dependencies.
        return None

    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")

# ------------------ ASK LLM ------------------
def ask_llm(prompt: str):
    provider = settings.LLM_PROVIDER.lower()

    # Use local Ollama CLI for local LLMs (no extra Python package required)
    if provider == "ollama":
        model = settings.OLLAMA_MODEL
        cli = settings.OLLAMA_CLI or "ollama"
        try:
            proc = subprocess.run(
                [cli, "run", model],
                input=prompt,
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                timeout=int(settings.OLLAMA_TIMEOUT or 300),
            )
            if proc.returncode != 0:
                raise RuntimeError(proc.stderr.strip() or "Ollama CLI error")
            return proc.stdout.strip()
        except subprocess.TimeoutExpired:
            # Retry once with an extended timeout
            extended = int((settings.OLLAMA_TIMEOUT or 300) * 2)
            try:
                proc = subprocess.run(
                    [cli, "run", model],
                    input=prompt,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    capture_output=True,
                    timeout=extended,
                )
                if proc.returncode != 0:
                    raise RuntimeError(proc.stderr.strip() or "Ollama CLI error on retry")
                return proc.stdout.strip()
            except Exception as e:
                raise RuntimeError(f"Ollama timeout after retry: {e}")
        except Exception as e:
            raise RuntimeError(f"Ollama error: {e}")

    if provider == "openrouter":
        api_key = settings.OPENROUTER_API_KEY
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not configured")

        payload = {
            "model": settings.OPENROUTER_MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        }

        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            settings.OPENROUTER_BASE_URL,
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost",
                "X-Title": "qdrant-ping-rag",
            },
            method="POST",
        )

        last_err = None
        for attempt in range(1, 6):
            try:
                with request.urlopen(req, timeout=120) as resp:
                    resp_payload = json.loads(resp.read().decode("utf-8"))
                last_err = None
                break
            except error.HTTPError as e:
                details = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else str(e)
                if e.code == 429 and attempt < 6:
                    time.sleep(min(2 ** attempt, 20))
                    last_err = RuntimeError(f"OpenRouter HTTP 429: {details}")
                    continue
                raise RuntimeError(f"OpenRouter HTTP {e.code}: {details}")
            except Exception as e:
                if attempt < 6:
                    time.sleep(min(2 ** attempt, 20))
                    last_err = RuntimeError(f"OpenRouter error: {e}")
                    continue
                raise RuntimeError(f"OpenRouter error: {e}")

        if last_err is not None:
            raise last_err

        choices = resp_payload.get("choices") or []
        if not choices:
            raise RuntimeError("OpenRouter returned no choices")
        message = (choices[0] or {}).get("message") or {}
        content = message.get("content")
        if not content:
            raise RuntimeError("OpenRouter returned empty content")
        return content.strip()

    # Fallback: use provider-specific client
    llm = get_llm()
    response = llm.invoke(prompt)
    return response.content


def _is_prereq_intent(question: str) -> bool:
    q = (question or "").lower()
    return any(
        marker in q
        for marker in [
            "can i take",
            "can i enroll",
            "enroll in",
            "can i register",
            "register for",
            "am i eligible",
            "eligible for",
            "prerequisite",
            "prerequisites",
        ]
    )


def _extract_target_course(question: str, q_codes):
    q = (question or "").upper()
    patterns = [
        rf"TAKE\s+({COURSE_CODE_REGEX})",
        rf"ELIGIBLE\s+FOR\s+({COURSE_CODE_REGEX})",
        rf"PREREQUISITES?\s+(?:OF|FOR)\s+({COURSE_CODE_REGEX})",
    ]
    for pat in patterns:
        match = re.search(pat, q)
        if match:
            return match.group(1)
    return q_codes[0] if q_codes else None


def _extract_completed_and_missing_courses(question: str, target_course: str):
    q_upper = (question or "").upper()
    all_codes = _extract_course_codes(q_upper)
    others = [c for c in all_codes if c != target_course]

    missing = set()
    completed = set()

    if any(k in q_upper for k in ["NOT COMPLETED", "HAVE NOT COMPLETED", "WITHOUT"]):
        missing.update(others)

    if "COMPLETED" in q_upper and not any(k in q_upper for k in ["NOT COMPLETED", "HAVE NOT COMPLETED"]):
        completed.update(others)

    return sorted(completed), sorted(missing)


def _collect_prereq_evidence(question: str, docs, target_course: str, observed_courses):
    """Collect and score prerequisite evidence from retrieved docs and a focused follow-up search."""
    candidates = list(docs)
    if target_course:
        focused_query = f"{target_course} prerequisites"
        try:
            focused = db.similarity_search_with_score(focused_query, k=12)
            candidates.extend(focused)
        except Exception:
            pass

    best_codes = []
    best_citation = None
    best_score = -1
    observed = set(observed_courses or [])
    matched_observed = set()

    for idx, (doc, _score) in enumerate(candidates):
        raw_text = (doc.page_content or "")
        text = raw_text.replace("\n", " ").strip()
        text_lower = raw_text.lower()
        if "prerequisite" not in text_lower:
            continue

        pre_idx = text_lower.find("prerequisite")
        pre_window = raw_text[pre_idx: pre_idx + 260] if pre_idx >= 0 else raw_text
        doc_codes = set(_extract_course_codes(pre_window))
        if target_course and target_course in doc_codes:
            doc_codes.discard(target_course)

        if not doc_codes:
            continue

        text_upper = raw_text.upper()
        target_before_prereq = 0.0
        target_after_prereq = 0.0
        if target_course and pre_idx >= 0:
            before = text_upper[:pre_idx]
            after = text_upper[pre_idx:]
            target_before_prereq = 2.5 if target_course in before else 0.0
            target_after_prereq = 0.5 if target_course in after else 0.0

        observed_overlap = len(observed & doc_codes)
        score = (2.0 * observed_overlap) + target_before_prereq + target_after_prereq + (1.0 / (idx + 1))

        if score > best_score:
            best_score = score
            best_citation = text
            best_codes = sorted(doc_codes)

        if observed_overlap > 0:
            matched_observed.update(observed & doc_codes)

    if best_citation and len(best_citation) > 300:
        best_citation = best_citation[:297] + "..."

    if matched_observed:
        return sorted(matched_observed), best_citation

    return best_codes, best_citation


def extract_prereq_decision(question: str, docs):
    """Structured prerequisite extraction for eligibility and prerequisite queries."""
    if not _is_prereq_intent(question):
        return None

    q_codes = _extract_course_codes(question)
    if not q_codes:
        return None

    target_course = _extract_target_course(question, q_codes)
    completed_courses, missing_courses = _extract_completed_and_missing_courses(question, target_course)

    observed_courses = sorted(set(completed_courses) | set(missing_courses))
    required_courses, citation = _collect_prereq_evidence(question, docs, target_course, observed_courses)
    if not required_courses:
        return {
            "applied": True,
            "target_course": target_course,
            "completed_courses": completed_courses,
            "missing_courses": missing_courses,
            "required_courses": [],
            "eligibility": None,
            "reason": "Could not find explicit prerequisite evidence in retrieved context.",
            "citation": None,
        }

    eligibility = None
    reason = f"{target_course} requires one of " + ", ".join(required_courses) + "."

    if missing_courses and set(required_courses).issubset(set(missing_courses)):
        eligibility = False
    elif completed_courses and set(completed_courses).intersection(set(required_courses)):
        eligibility = True
    elif is_eligibility_query(question):
        reason += " Provided completion details are insufficient for a deterministic eligibility decision."

    return {
        "applied": True,
        "target_course": target_course,
        "completed_courses": completed_courses,
        "missing_courses": missing_courses,
        "required_courses": required_courses,
        "eligibility": eligibility,
        "reason": reason,
        "citation": citation,
    }


def is_eligibility_query(question: str) -> bool:
    q = (question or "").lower()
    return any(
        marker in q
        for marker in [
            "can i take",
            "can i enroll",
            "enroll in",
            "can i register",
            "register for",
            "am i eligible",
            "eligible for",
        ]
    )


def apply_user_context_to_prereq_decision(decision: dict | None, user_context: dict | None):
    """Augment prerequisite decision with explicit user-provided context."""
    if not decision or not decision.get("applied"):
        return decision

    ctx = user_context or {}
    completed = set(decision.get("completed_courses") or [])
    completed.update({c.upper() for c in (ctx.get("completed_courses") or []) if isinstance(c, str)})
    missing = set(decision.get("missing_courses") or [])

    required = set(decision.get("required_courses") or [])

    decision["completed_courses"] = sorted(completed)

    if required and completed.intersection(required):
        decision["eligibility"] = True
        decision["reason"] = (
            f"{decision.get('target_course')} requires one of "
            + ", ".join(sorted(required))
            + ". You have completed "
            + ", ".join(sorted(completed.intersection(required)))
            + "."
        )
    elif required and missing and required.issubset(missing):
        decision["eligibility"] = False
        decision["reason"] = (
            f"{decision.get('target_course')} requires one of "
            + ", ".join(sorted(required))
            + ". You reported not completing these prerequisite options."
        )

    return decision


def build_clarification_payload(decision: dict | None, question: str, user_context: dict | None = None):
    """Return follow-up questions if eligibility intent is detected but key details are missing."""
    if not decision or not decision.get("applied"):
        return None

    if not is_eligibility_query(question):
        return None

    ctx = user_context or {}
    completed_courses = ctx.get("completed_courses") or decision.get("completed_courses") or []
    completed_courses_declared = bool(ctx.get("completed_courses_declared"))
    gpa = ctx.get("gpa")
    q = (question or "").lower()
    asks_for_grade = any(token in q for token in ["gpa", "grade", "cgpa", "grade point average"])

    missing_fields = []
    follow_ups = []

    if not completed_courses and not completed_courses_declared:
        missing_fields.append("completed_courses")
        follow_ups.append(
            {
                "field": "completed_courses",
                "question": "Which courses have you already completed? Please provide course codes like AH110, AH152.",
            }
        )

    if asks_for_grade and gpa in (None, ""):
        missing_fields.append("gpa")
        follow_ups.append(
            {
                "field": "gpa",
                "question": "What is your current GPA?",
            }
        )

    if not missing_fields:
        return None

    return {
        "needs_clarification": True,
        "message": "I need a few details before I can give a definitive eligibility answer.",
        "missing_fields": missing_fields,
        "follow_up_questions": follow_ups,
        "example_context": {
            "completed_courses": ["AH110"],
            "semester": "fall",
            "gpa": 3.4,
        },
    }


def extract_user_context_from_text(text: str | None):
    """Parse free-text user follow-up into structured fields used by eligibility logic."""
    raw = (text or "").strip()
    if not raw:
        return {}

    upper = raw.upper()
    lower = raw.lower()

    completed_courses = set()
    currently_enrolled_courses = set()
    completed_courses_declared = False
    currently_enrolled_declared = False

    for m in re.finditer(r"(?:completed|completing|passed|finished|done)\s+([^\.;\n]+)", raw, flags=re.IGNORECASE):
        # Check if preceded by negation (have not, haven't, did not, no, etc.)
        preceding_text = raw[:m.start()].lower()
        negation_patterns = [r"have\s+not\s+$", r"haven't\s+$", r"did\s+not\s+$", r"didn't\s+$", r"not\s+$", r"no\s+$"]
        if any(re.search(pat, preceding_text) for pat in negation_patterns):
            continue  # Skip this match - it's a negated completion
        
        segment = m.group(1)
        # Prevent enrolled/taking clauses from being counted as completed.
        segment = re.split(r"\b(?:currently\s+enrolled|enrolled|taking|instructor\s+consent|consent\s+of\s+instructor)\b", segment, maxsplit=1, flags=re.IGNORECASE)[0]
        completed_courses.update(_extract_course_codes(segment))

    explicit_no_completed = any(
        re.search(pat, lower)
        for pat in [
            r"\bno\s+courses\b",
            r"\bno\s+course\s+codes\b",
            r"\bnone\s+completed\b",
            r"\bnot\s+completed\s+any\b",
            r"\bhaven'?t\s+completed\b",
            r"\bhave\s+not\s+completed\b",
            r"\bnope\b",
        ]
    )

    # Mark completed_courses_declared if:
    # 1. User provided actual courses, OR
    # 2. For now, don't declare based on explicit negations in questions
    #    (let build_clarification_payload handle this based on whether
    #     the question's mentioned courses are actual prerequisites)
    if completed_courses:
        completed_courses_declared = True

    for m in re.finditer(r"(?:currently\s+enrolled|enrolled|taking)\s+(?:in\s+)?([^\.;\n]+)", raw, flags=re.IGNORECASE):
        # Check if preceded by negation (not enrolled, haven't taken, etc.)
        preceding_text = raw[:m.start()].lower()
        negation_patterns = [r"not\s+$", r"haven't\s+$", r"have\s+not\s+$", r"didn't\s+$", r"did\s+not\s+$", r"no\s+$"]
        if any(re.search(pat, preceding_text) for pat in negation_patterns):
            continue  # Skip this match - it's a negated enrollment
        
        segment = m.group(1)
        currently_enrolled_courses.update(_extract_course_codes(segment))

    explicit_not_enrolled = any(
        re.search(pat, lower)
        for pat in [
            r"\bnot\s+(?:currently\s+)?enrolled\b",
            r"\bnot\s+taking\b",
            r"\bno\s+co-?requisites?\b",
            r"\bnone\b.*\b(co-?requisites?|enrolled|taking)\b",
        ]
    )

    # Mark currently_enrolled_declared if:
    # 1. User provided actual courses, OR  
    # 2. For now, don't declare based on explicit negations in questions
    #    (let build_clarification_payload handle this based on whether
    #     the question's mentioned courses are actual co-requisites)
    if currently_enrolled_courses:
        currently_enrolled_declared = True

    # Fallback only when no qualifier markers are present.
    has_qualifiers = any(
        k in lower
        for k in [
            "completed",
            "passed",
            "finished",
            "done",
            "enrolled",
            "taking",
            "instructor consent",
            "consent of instructor",
        ]
    )
    if not completed_courses and not currently_enrolled_courses and not has_qualifiers:
        bare_codes = _extract_course_codes(raw)
        if bare_codes:
            completed_courses.update(bare_codes)

    semester = None
    if "autumn" in lower or "fall" in lower:
        semester = "fall"
    elif "spring" in lower:
        semester = "spring"
    elif "summer" in lower:
        semester = "summer"
    elif "winter" in lower:
        semester = "winter"

    gpa = None
    gpa_patterns = [
        r"\bgpa\s*[:=]?\s*(\d(?:\.\d{1,2})?)\b",
        r"\bgrade\s*point\s*average\s*[:=]?\s*(\d(?:\.\d{1,2})?)\b",
    ]
    for pat in gpa_patterns:
        match = re.search(pat, lower)
        if match:
            try:
                parsed = float(match.group(1))
                if 0.0 <= parsed <= 4.0:
                    gpa = parsed
                    break
            except ValueError:
                pass

    # Fallback: allow a plain 0-4 decimal when strongly indicated by nearby words.
    if gpa is None and any(k in lower for k in ["gpa", "grade", "cgpa"]):
        plain = re.search(r"\b(\d(?:\.\d{1,2})?)\b", lower)
        if plain:
            try:
                parsed = float(plain.group(1))
                if 0.0 <= parsed <= 4.0:
                    gpa = parsed
            except ValueError:
                pass

    # Parse explicit per-course grades: "AH110 grade B+"
    grades = {}
    for m in re.finditer(rf"\b({COURSE_CODE_REGEX})\b[^\n\.]{0,30}?\bgrade\b[^\n\.]{0,10}?(A\+|A|A-|B\+|B|B-|C\+|C|C-|D\+|D|D-|F)\b", raw, flags=re.IGNORECASE):
        grades[m.group(1).upper()] = m.group(2).upper()

    # Parse currently enrolled courses from wording like "enrolled in AH152"
    instructor_consent_for = []
    if "instructor consent" in lower or "consent of instructor" in lower:
        instructor_consent_for = _extract_course_codes(raw)

    parsed_context = {}
    if completed_courses:
        parsed_context["completed_courses"] = sorted(completed_courses)
    if completed_courses_declared:
        parsed_context["completed_courses_declared"] = True
    if currently_enrolled_courses:
        parsed_context["currently_enrolled_courses"] = sorted(currently_enrolled_courses)
    if currently_enrolled_declared:
        parsed_context["currently_enrolled_declared"] = True
    if semester:
        parsed_context["semester"] = semester
    if gpa is not None:
        parsed_context["gpa"] = gpa
    if grades:
        parsed_context["grades"] = grades
    if instructor_consent_for:
        parsed_context["instructor_consent_for"] = instructor_consent_for

    return parsed_context


def merge_user_context(primary: dict | None, secondary: dict | None):
    """Merge user contexts, preserving explicit fields while combining completed_courses."""
    p = dict(primary or {})
    s = dict(secondary or {})

    merged = {}
    merged.update(p)
    merged.update({k: v for k, v in s.items() if v not in (None, "", [])})

    p_courses = {c.upper() for c in (p.get("completed_courses") or []) if isinstance(c, str)}
    s_courses = {c.upper() for c in (s.get("completed_courses") or []) if isinstance(c, str)}
    all_courses = sorted(p_courses | s_courses)
    if all_courses:
        merged["completed_courses"] = all_courses

    p_enrolled = {c.upper() for c in (p.get("currently_enrolled_courses") or []) if isinstance(c, str)}
    s_enrolled = {c.upper() for c in (s.get("currently_enrolled_courses") or []) if isinstance(c, str)}
    all_enrolled = sorted(p_enrolled | s_enrolled)
    if all_enrolled:
        merged["currently_enrolled_courses"] = all_enrolled

    p_consent = {c.upper() for c in (p.get("instructor_consent_for") or []) if isinstance(c, str)}
    s_consent = {c.upper() for c in (s.get("instructor_consent_for") or []) if isinstance(c, str)}
    all_consent = sorted(p_consent | s_consent)
    if all_consent:
        merged["instructor_consent_for"] = all_consent

    p_grades = {k.upper(): str(v).upper() for k, v in (p.get("grades") or {}).items()}
    s_grades = {k.upper(): str(v).upper() for k, v in (s.get("grades") or {}).items()}
    merged_grades = {}
    merged_grades.update(p_grades)
    merged_grades.update(s_grades)
    if merged_grades:
        merged["grades"] = merged_grades

    if bool(p.get("completed_courses_declared")) or bool(s.get("completed_courses_declared")):
        merged["completed_courses_declared"] = True

    if bool(p.get("currently_enrolled_declared")) or bool(s.get("currently_enrolled_declared")):
        merged["currently_enrolled_declared"] = True

    return merged


def build_answer_from_prereq_decision(decision: dict | None):
    if not decision or not decision.get("applied"):
        return None

    if decision.get("eligibility") is False:
        return (
            "Final answer: Not eligible\n"
            f"Why: {decision.get('reason')}\n"
            f"Citations: \"{decision.get('citation')}\""
        )

    if decision.get("eligibility") is True:
        return (
            "Final answer: Eligible\n"
            f"Why: {decision.get('reason')}\n"
            f"Citations: \"{decision.get('citation')}\""
        )

    if decision.get("required_courses"):
        return (
            "Final answer: Prerequisites identified\n"
            f"Why: {decision.get('reason')}\n"
            f"Citations: \"{decision.get('citation')}\""
        )

    return None
