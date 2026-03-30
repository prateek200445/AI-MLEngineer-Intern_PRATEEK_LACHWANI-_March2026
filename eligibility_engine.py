from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Callable, Dict, List, Tuple


COURSE_CODE_REGEX = r"[A-Z]{1,2}\d{2,3}[A-Z]?"
COURSE_CODE_PATTERN = re.compile(rf"\b{COURSE_CODE_REGEX}\b")
GRADE_PATTERN = re.compile(r"\b(A\+|A|A-|B\+|B|B-|C\+|C|C-|D\+|D|D-|F)\b", re.IGNORECASE)


GRADE_POINTS = {
    "A+": 4.0,
    "A": 4.0,
    "A-": 3.7,
    "B+": 3.3,
    "B": 3.0,
    "B-": 2.7,
    "C+": 2.3,
    "C": 2.0,
    "C-": 1.7,
    "D+": 1.3,
    "D": 1.0,
    "D-": 0.7,
    "F": 0.0,
}


@dataclass
class RuleProfile:
    target_course: str
    any_of: List[str] = field(default_factory=list)
    all_of: List[str] = field(default_factory=list)
    co_requisites: List[str] = field(default_factory=list)
    min_grade_by_course: Dict[str, str] = field(default_factory=dict)
    allows_instructor_consent: bool = False
    prerequisite_chain: Dict[str, List[str]] = field(default_factory=dict)
    evidence: List[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class EligibilityResult:
    decision: str  # eligible | not_eligible | needs_more_info | unknown
    reasons: List[str]
    missing_requirements: List[str]
    missing_inputs: List[str]


def _next_term(term: str) -> str:
    order = ["fall", "spring", "summer"]
    t = (term or "fall").lower()
    if t not in order:
        t = "fall"
    return order[(order.index(t) + 1) % len(order)]


def _dedupe_keep_order(items: List[str]) -> List[str]:
    return list(dict.fromkeys(items))


def extract_course_codes(text: str | None) -> List[str]:
    return list(dict.fromkeys(COURSE_CODE_PATTERN.findall((text or "").upper())))


def extract_target_course(question: str | None) -> str | None:
    q = (question or "").upper()
    patterns = [
        rf"TAKE\s+({COURSE_CODE_REGEX})",
        rf"ELIGIBLE\s+FOR\s+({COURSE_CODE_REGEX})",
        rf"PREREQUISITES?\s+(?:OF|FOR)\s+({COURSE_CODE_REGEX})",
    ]
    for pat in patterns:
        m = re.search(pat, q)
        if m:
            return m.group(1)

    codes = extract_course_codes(q)
    return codes[0] if codes else None


def _parse_prereq_codes(target_course: str, text: str) -> Tuple[List[str], List[str], bool]:
    lower = text.lower()
    idx = lower.find("prerequisite")
    if idx < 0:
        return [], [], False

    window = text[idx: idx + 320]
    codes = [c for c in extract_course_codes(window) if c != target_course]
    if not codes:
        return [], [], False

    low_window = window.lower()
    has_or = " or " in low_window
    has_and = " and " in low_window

    # Mixed wording is treated conservatively as any_of for now.
    if has_and and not has_or:
        return [], sorted(set(codes)), True
    return sorted(set(codes)), [], True


def _parse_coreq_codes(target_course: str, text: str) -> List[str]:
    lower = text.lower()
    idx = lower.find("co-requisite")
    if idx < 0:
        idx = lower.find("corequisite")
    if idx < 0:
        return []

    window = text[idx: idx + 220]
    return sorted({c for c in extract_course_codes(window) if c != target_course})


def _parse_grade_requirements(text: str) -> Dict[str, str]:
    results: Dict[str, str] = {}
    lower = text.lower()

    # Pattern: AH110 minimum grade of B
    p1 = re.finditer(
        rf"\b({COURSE_CODE_REGEX})\b[^\n\.]{0,80}?(minimum\s+grade\s+of|grade\s+of\s+at\s+least)\s+(A\+|A|A-|B\+|B|B-|C\+|C|C-|D\+|D|D-|F)",
        text,
        flags=re.IGNORECASE,
    )
    for m in p1:
        results[m.group(1).upper()] = m.group(3).upper()

    # Pattern: minimum grade of B in AH110
    p2 = re.finditer(
        rf"(minimum\s+grade\s+of|grade\s+of\s+at\s+least)\s+(A\+|A|A-|B\+|B|B-|C\+|C|C-|D\+|D|D-|F)[^\n\.]{0,80}?\b({COURSE_CODE_REGEX})\b",
        text,
        flags=re.IGNORECASE,
    )
    for m in p2:
        results[m.group(3).upper()] = m.group(2).upper()

    if "minimum grade" in lower and not results:
        # Fallback to avoid missing obvious grade lines due to format noise.
        codes = extract_course_codes(text)
        g = GRADE_PATTERN.search(text)
        if codes and g:
            results[codes[0]] = g.group(1).upper()

    return results


def build_rule_profile(
    target_course: str,
    docs: List[Tuple[object, float]],
    search_fn: Callable[[str, int], List[Tuple[object, float]]] | None = None,
    depth: int = 2,
) -> RuleProfile:
    profile = RuleProfile(target_course=target_course)

    candidates = list(docs)
    if search_fn:
        try:
            candidates.extend(search_fn(f"{target_course} prerequisites", 12))
            candidates.extend(search_fn(f"{target_course} co-requisites", 8))
            candidates.extend(search_fn(f"{target_course} minimum grade", 8))
            candidates.extend(search_fn(f"{target_course} instructor consent", 8))
        except Exception:
            pass

    evidence_count = 0
    for doc, _score in candidates:
        text = (getattr(doc, "page_content", "") or "").replace("\n", " ").strip()
        if not text:
            continue

        low = text.lower()
        any_of, all_of, has_prereq = _parse_prereq_codes(target_course, text)
        coreq = _parse_coreq_codes(target_course, text)
        grades = _parse_grade_requirements(text)
        has_consent = "instructor consent" in low or "consent of instructor" in low

        if has_prereq:
            profile.any_of = sorted(set(profile.any_of) | set(any_of))
            profile.all_of = sorted(set(profile.all_of) | set(all_of))
            evidence_count += 1
        if coreq:
            profile.co_requisites = sorted(set(profile.co_requisites) | set(coreq))
            evidence_count += 1
        if grades:
            profile.min_grade_by_course.update(grades)
            evidence_count += 1
        if has_consent:
            profile.allows_instructor_consent = True
            evidence_count += 1

        if has_prereq or coreq or grades or has_consent:
            profile.evidence.append(text[:300])

    # Build simple prerequisite chain (A -> B -> C) with one focused hop per prerequisite.
    if search_fn and depth > 1:
        for pre in profile.any_of + profile.all_of:
            try:
                child_docs = search_fn(f"{pre} prerequisites", 6)
                child_any, child_all = [], []
                for child_doc, _ in child_docs:
                    t = (getattr(child_doc, "page_content", "") or "")
                    a2, b2, has = _parse_prereq_codes(pre, t)
                    if has:
                        child_any.extend(a2)
                        child_all.extend(b2)
                chain_codes = sorted(set(child_any + child_all))
                if chain_codes:
                    profile.prerequisite_chain[pre] = chain_codes
            except Exception:
                continue

    profile.confidence = min(1.0, 0.2 + 0.15 * evidence_count)
    return profile


def profile_to_dict(profile: RuleProfile) -> dict:
    return {
        "target_course": profile.target_course,
        "any_of": _dedupe_keep_order(profile.any_of),
        "all_of": _dedupe_keep_order(profile.all_of),
        "co_requisites": _dedupe_keep_order(profile.co_requisites),
        "min_grade_by_course": profile.min_grade_by_course,
        "allows_instructor_consent": profile.allows_instructor_consent,
        "prerequisite_chain": profile.prerequisite_chain,
        "evidence": profile.evidence,
        "confidence": profile.confidence,
    }


def profile_from_dict(data: dict) -> RuleProfile:
    return RuleProfile(
        target_course=data.get("target_course", ""),
        any_of=data.get("any_of", []) or [],
        all_of=data.get("all_of", []) or [],
        co_requisites=data.get("co_requisites", []) or [],
        min_grade_by_course=data.get("min_grade_by_course", {}) or {},
        allows_instructor_consent=bool(data.get("allows_instructor_consent", False)),
        prerequisite_chain=data.get("prerequisite_chain", {}) or {},
        evidence=data.get("evidence", []) or [],
        confidence=float(data.get("confidence", 0.0) or 0.0),
    )


def load_rule_store(path: str) -> Dict[str, dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}

    courses = data.get("courses") if isinstance(data, dict) else None
    if not isinstance(courses, dict):
        return {}

    normalized = {}
    for k, v in courses.items():
        if not isinstance(v, dict):
            continue
        normalized[str(k).upper()] = v
    return normalized


def get_profile_from_rule_store(target_course: str, store: Dict[str, dict]) -> RuleProfile | None:
    if not target_course or not store:
        return None
    data = store.get(target_course.upper())
    if not data:
        return None
    profile = profile_from_dict(data)
    if not profile.target_course:
        profile.target_course = target_course.upper()
    return profile


def _grade_meets(actual: str, required: str) -> bool:
    a = GRADE_POINTS.get((actual or "").upper())
    r = GRADE_POINTS.get((required or "").upper())
    if a is None or r is None:
        return False
    return a >= r


def evaluate_profile(profile: RuleProfile, user_context: dict | None) -> EligibilityResult:
    ctx = user_context or {}
    completed = {c.upper() for c in (ctx.get("completed_courses") or []) if isinstance(c, str)}
    enrolled = {c.upper() for c in (ctx.get("currently_enrolled_courses") or []) if isinstance(c, str)}
    enrolled_declared = bool(ctx.get("currently_enrolled_declared"))
    consent_for = {c.upper() for c in (ctx.get("instructor_consent_for") or []) if isinstance(c, str)}
    grades = {k.upper(): str(v).upper() for k, v in (ctx.get("grades") or {}).items()}

    reasons: List[str] = []
    missing_reqs: List[str] = []
    missing_inputs: List[str] = []

    if profile.target_course in consent_for and profile.allows_instructor_consent:
        reasons.append("Instructor consent override is present.")
        return EligibilityResult("eligible", reasons, missing_reqs, missing_inputs)

    # either/or
    if profile.any_of:
        if not completed.intersection(set(profile.any_of)):
            missing_reqs.append("Complete at least one of: " + ", ".join(profile.any_of))

    # all-of
    if profile.all_of:
        missing_all = [c for c in profile.all_of if c not in completed]
        if missing_all:
            missing_reqs.append("Complete required prerequisites: " + ", ".join(missing_all))

    # co-req (completed or currently enrolled)
    if profile.co_requisites:
        unsatisfied = [c for c in profile.co_requisites if c not in completed and c not in enrolled]
        if unsatisfied:
            missing_reqs.append("Co-requisites must be completed or enrolled: " + ", ".join(unsatisfied))
            if not enrolled and not enrolled_declared:
                missing_inputs.append("currently_enrolled_courses")

    # min grades
    for c, required_grade in profile.min_grade_by_course.items():
        if c in completed:
            actual_grade = grades.get(c)
            if not actual_grade:
                missing_inputs.append("grades")
                missing_reqs.append(f"Need grade for {c} (minimum required: {required_grade}).")
            elif not _grade_meets(actual_grade, required_grade):
                missing_reqs.append(f"{c} requires minimum grade {required_grade}; current grade is {actual_grade}.")

    if missing_reqs:
        decision = "needs_more_info" if missing_inputs else "not_eligible"
        reasons.append("Eligibility constraints are not fully satisfied yet.")
        return EligibilityResult(decision, reasons, missing_reqs, sorted(set(missing_inputs)))

    reasons.append("All detected deterministic constraints are satisfied.")
    return EligibilityResult("eligible", reasons, missing_reqs, missing_inputs)


def _is_course_unlocked(profile: RuleProfile, completed: set[str]) -> bool:
    if profile.any_of and not completed.intersection(set(profile.any_of)):
        return False
    if profile.all_of and not set(profile.all_of).issubset(completed):
        return False
    return True


def build_term_plan(
    rule_store: Dict[str, dict],
    completed_courses: List[str] | None,
    target_course: str | None = None,
    target_prefix: str | None = None,
    start_term: str = "fall",
    max_courses_per_term: int = 3,
    max_credits: int = 12,
    term_count: int = 4,
) -> dict:
    completed = {c.upper() for c in (completed_courses or []) if isinstance(c, str)}

    target_course = (target_course or "").upper() or None
    target_prefix = (target_prefix or "").upper() or None

    prioritized: List[str] = []
    target_prereqs: List[str] = []
    
    if target_course and target_course in rule_store and target_course not in completed:
        target_profile = profile_from_dict(rule_store.get(target_course, {}))
        target_prereqs = (target_profile.all_of or []) + (target_profile.any_of or [])
        target_prereqs = [c for c in sorted(set(target_prereqs)) if c not in completed]
        if target_prereqs:
            prioritized.extend(target_prereqs)
        prioritized.append(target_course)
        target_prefix = target_course[:2]

    if target_prefix:
        prioritized.extend(
            [
                code
                for code in sorted(rule_store.keys())
                if code not in completed and code.startswith(target_prefix) and code not in prioritized
            ]
        )

    if not prioritized:
        prefixes = {c[:2] for c in completed if len(c) >= 2}
        prioritized = [
            code for code in sorted(rule_store.keys())
            if code not in completed and (code[:2] in prefixes)
        ]

    if not prioritized:
        prioritized = [code for code in sorted(rule_store.keys()) if code not in completed]

    candidates = _dedupe_keep_order(prioritized + [code for code in sorted(rule_store.keys()) if code not in completed])

    planned = set()
    terms = []
    current_term = (start_term or "fall").lower()

    for _ in range(max(1, term_count)):
        unlocked = []
        for code in candidates:
            if code in planned or code in completed:
                continue
            profile = profile_from_dict(rule_store.get(code, {}))
            if not profile.target_course:
                profile.target_course = code
            if _is_course_unlocked(profile, completed):
                unlocked.append((code, profile))

        picks = unlocked[:max(1, max_courses_per_term)]
        if not picks:
            terms.append({"term": current_term, "courses": [], "note": "No unlocked courses from current rule set."})
            current_term = _next_term(current_term)
            continue

        term_courses = []
        used_credits = 0
        for code, profile in picks:
            # Catalog currently lacks a canonical per-course credit map; default to 4.
            course_credits = 4
            if used_credits + course_credits > max(1, max_credits):
                continue

            satisfied_by = []
            if profile.any_of:
                satisfied_by.extend(sorted(completed.intersection(set(profile.any_of))))
            if profile.all_of:
                satisfied_by.extend([c for c in profile.all_of if c in completed])
            term_courses.append(
                {
                    "course": code,
                    "credits": course_credits,
                    "why_unlocked": {
                        "any_of": profile.any_of,
                        "all_of": profile.all_of,
                        "satisfied_by": sorted(set(satisfied_by)),
                    },
                }
            )
            planned.add(code)
            used_credits += course_credits

        completed.update([c["course"] for c in term_courses])
        terms.append({"term": current_term, "courses": term_courses, "total_credits": used_credits})
        current_term = _next_term(current_term)

    return {
        "status": "ok",
        "start_term": start_term,
        "max_courses_per_term": max_courses_per_term,
        "max_credits": max_credits,
        "term_count": term_count,
        "planned_terms": terms,
        "risks_assumptions": [
            "Course offering availability by semester is not guaranteed by current documents.",
            "Credits per course defaulted to 4 when not explicitly available in policy text.",
            "Advisor validation is recommended before final enrollment.",
        ],
    }
