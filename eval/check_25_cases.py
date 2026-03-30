import json
import re
import os
import time
from pathlib import Path

import requests


QUES_FILE = Path(r"d:\qdrant_ping\ques")
API_URL = os.getenv("RAG_API_URL", "http://127.0.0.1:8000/query")


def parse_cases(text: str):
    lines = text.splitlines()
    cases = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        m = re.match(r"^(\d+)\.\s+(.*)$", line)
        if not m:
            i += 1
            continue

        idx = int(m.group(1))
        question = m.group(2).strip()

        expected = None
        j = i + 1
        while j < len(lines):
            if re.match(r"^\d+\.\s+", lines[j].strip()):
                break
            if lines[j].strip().lower() == "answer / plan:":
                k = j + 1
                while k < len(lines) and not lines[k].strip():
                    k += 1
                if k < len(lines):
                    expected = lines[k].strip()
                break
            j += 1

        if expected:
            cases.append({"id": idx, "question": question, "expected_plan": expected})
        i = j
    return cases


def normalize_expected(expected_plan: str):
    s = expected_plan.lower()
    if "need more information" in s:
        return "need_more_info"
    if "not eligible" in s:
        return "not_eligible"
    if "i don\'t have" in s or "not available" in s or "not specified" in s:
        return "abstain"
    return "policy_answer"


def check_case(case):
    payload = {"question": case["question"]}
    data = None
    last_error = None
    for attempt in range(1, 4):
        try:
            r = requests.post(API_URL, json=payload, timeout=120)
            r.raise_for_status()
            data = r.json()
            last_error = None
            break
        except Exception as e:
            last_error = e
            if attempt < 3:
                time.sleep(5 * attempt)

    if data is None:
        return {
            "id": case["id"],
            "question": case["question"],
            "expected_plan": case["expected_plan"],
            "expected_class": normalize_expected(case["expected_plan"]),
            "status": "error",
            "error": str(last_error),
        }

    answer = (data.get("answer") or "").strip()
    decision = data.get("decision")
    expected_class = normalize_expected(case["expected_plan"])

    if expected_class == "policy_answer":
        matched = bool(answer) and "i don't have that information" not in answer.lower()
    elif expected_class == "abstain":
        matched = (
            decision == "abstain"
            or "i don't have that information" in answer.lower()
            or "not available" in answer.lower()
            or "not specified" in answer.lower()
        )
    else:
        matched = decision == expected_class

    return {
        "id": case["id"],
        "question": case["question"],
        "expected_plan": case["expected_plan"],
        "expected_class": expected_class,
        "actual_decision": decision,
        "actual_answer": answer[:240],
        "matched": matched,
        "status": "ok",
    }


def main():
    text = QUES_FILE.read_text(encoding="utf-8")
    cases = parse_cases(text)
    cases = [c for c in cases if 1 <= c["id"] <= 26]

    results = [check_case(case) for case in cases]
    mismatches = [r for r in results if r.get("status") == "ok" and not r.get("matched")]
    errors = [r for r in results if r.get("status") == "error"]

    out_path = Path(r"d:\qdrant_ping\eval\check_25_results.json")
    out_path.write_text(json.dumps({
        "total": len(results),
        "mismatches": len(mismatches),
        "errors": len(errors),
        "results": results,
    }, indent=2), encoding="utf-8")

    print(f"Total: {len(results)}")
    print(f"Mismatches: {len(mismatches)}")
    print(f"Errors: {len(errors)}")
    if mismatches:
        print("\nMISMATCH CASE IDS:", ", ".join(str(m["id"]) for m in mismatches))
    if errors:
        print("\nERROR CASE IDS:", ", ".join(str(e["id"]) for e in errors))
    print(f"Saved detailed results to: {out_path}")


if __name__ == "__main__":
    main()
