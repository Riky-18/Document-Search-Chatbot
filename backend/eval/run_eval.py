#!/usr/bin/env python3
"""Evaluation harness for AI Document Chatbot RAG pipeline.

Measures retrieval hit rate (via retrieved chunks), refusal despite retrieval hit,
keyword answer accuracy, unanswerable refusal rate, and latency percentiles
across verified evaluation questions.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
import uuid

# Configurable list of refusal phrases indicating the bot acknowledged missing info
REFUSAL_PHRASES = [
    "don't know",
    "dont know",
    "do not know",
    "not mentioned in the context",
    "not provided in the context",
    "not found in the context",
    "not contain information",
    "context does not contain",
    "context does not mention",
    "context does not provide",
    "no information",
    "not in the context",
    "no relevant information",
    "not in the provided",
]


def _safe_print(text: str = "") -> None:
    """Safely print text avoiding Unicode encoding crashes on Windows consoles."""
    try:
        print(text, flush=True)
    except (UnicodeEncodeError, OSError):
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        try:
            sys.stdout.buffer.write(text.encode(encoding, errors="replace") + b"\n")
            sys.stdout.flush()
        except Exception:
            print(text.encode("ascii", errors="backslashreplace").decode("ascii"), flush=True)


def parse_retry_after(err: urllib.error.HTTPError, default_wait: float = 5.0) -> float:
    """Extract retry delay from headers or body if available."""
    retry_header = err.headers.get("Retry-After")
    if retry_header:
        try:
            return float(retry_header) + 1.0
        except ValueError:
            pass
    try:
        body = err.read().decode("utf-8", errors="replace")
        match = re.search(r"retry\s+(?:in|after)\s+([0-9]+(?:\.[0-9]+)?)\s*(?:s|sec|seconds)?", body, re.I)
        if match:
            return float(match.group(1)) + 1.0
    except Exception:
        pass
    return default_wait


def request_with_retry(
    req_factory,
    max_retries: int = 3,
    base_wait: float = 5.0,
    action_desc: str = "request",
) -> dict:
    """Execute a urllib request callable with exponential backoff on HTTP 429."""
    for attempt in range(max_retries + 1):
        try:
            req = req_factory()
            with urllib.request.urlopen(req) as resp:
                data = resp.read().decode("utf-8")
                return json.loads(data) if data else {}
        except urllib.error.HTTPError as err:
            if err.code == 429 and attempt < max_retries:
                wait_sec = parse_retry_after(err, base_wait * (2 ** attempt))
                _safe_print(f"  [429 Rate Limit] Retrying {action_desc} in {wait_sec:.1f}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait_sec)
            else:
                err_body = ""
                try:
                    err_body = err.read().decode("utf-8", errors="replace")
                except Exception:
                    pass
                raise RuntimeError(f"HTTP {err.code} on {action_desc}: {err_body or err.reason}") from err
        except urllib.error.URLError as err:
            raise RuntimeError(f"Network error on {action_desc}: {err.reason}") from err


def get_stats(api_url: str, timeout: float = 2.0) -> dict:
    """GET /stats: fetch document statistics and top_k configuration."""
    url = f"{api_url.rstrip('/')}/stats"
    req = urllib.request.Request(url, headers={}, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read().decode("utf-8")
        return json.loads(data) if data else {}


def clear_index(api_url: str) -> dict:
    """POST /clear: reset the vector store index."""
    url = f"{api_url.rstrip('/')}/clear"
    def _make_req():
        return urllib.request.Request(url, data=b"", headers={}, method="POST")
    return request_with_retry(_make_req, action_desc="POST /clear")


def upload_pdf(api_url: str, pdf_path: Path) -> dict:
    """POST /upload: upload a single PDF file using multipart/form-data."""
    url = f"{api_url.rstrip('/')}/upload"
    boundary = f"----EvalBoundary{uuid.uuid4().hex}"
    filename = pdf_path.name
    with open(pdf_path, "rb") as f:
        file_bytes = f.read()

    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: application/pdf\r\n\r\n"
    ).encode("utf-8") + file_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")

    def _make_req():
        return urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Content-Length": str(len(body)),
            },
            method="POST",
        )

    return request_with_retry(_make_req, action_desc=f"POST /upload ({filename})")


def ask_question(api_url: str, question: str) -> dict:
    """POST /ask: query the RAG pipeline."""
    url = f"{api_url.rstrip('/')}/ask"
    payload = json.dumps({"question": question}).encode("utf-8")

    def _make_req():
        return urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

    return request_with_retry(_make_req, action_desc="POST /ask")


def is_refusal(answer: str) -> bool:
    """Check if the answer text contains any recognized refusal phrase."""
    lower = answer.lower()
    return any(phrase in lower for phrase in REFUSAL_PHRASES)


def calculate_p95(values: list[float]) -> float:
    """Calculate the 95th percentile of a list of floats."""
    if not values:
        return 0.0
    sorted_v = sorted(values)
    idx = int(0.95 * len(sorted_v))
    return sorted_v[min(idx, len(sorted_v) - 1)]


def find_test_pdfs_dir(override: str | None = None) -> Path:
    """Locate the test_pdfs directory."""
    if override:
        p = Path(override).resolve()
        if p.exists() and p.is_dir():
            return p
        raise FileNotFoundError(f"Specified PDF directory does not exist: {override}")

    candidates = [
        Path("test_pdfs").resolve(),
        Path("../test_pdfs").resolve(),
        Path("../../test_pdfs").resolve(),
        Path(__file__).resolve().parent.parent.parent / "test_pdfs",
    ]
    for c in candidates:
        if c.exists() and c.is_dir():
            return c
    raise FileNotFoundError("Could not locate test_pdfs/ folder. Specify --pdf-dir.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run RAG evaluation harness.")
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="Base URL of running FastAPI backend (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=4.0,
        help="Sleep delay in seconds between questions to respect free-tier rate limits (default: 4.0)",
    )
    parser.add_argument(
        "--pdf-dir",
        default=None,
        help="Path to folder containing test PDFs (default: auto-detect test_pdfs)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all questions regardless of verified flag (for testing)",
    )

    args = parser.parse_args(argv)

    eval_dir = Path(__file__).resolve().parent
    questions_file = eval_dir / "questions.json"
    results_dir = eval_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    if not questions_file.exists():
        _safe_print(f"Error: {questions_file} not found.")
        return 1

    with open(questions_file, "r", encoding="utf-8") as f:
        all_questions = json.load(f)

    # Filter for verified items unless --all flag is passed
    if args.all:
        questions_to_run = all_questions
        _safe_print(f"Notice: --all flag specified. Running all {len(questions_to_run)} questions.")
    else:
        questions_to_run = [q for q in all_questions if q.get("verified") is True]

    # Query initial /stats to discover TOP_K if backend is running
    top_k_display = "unknown"
    try:
        initial_stats = get_stats(args.api_url)
        top_k_display = initial_stats.get("top_k", "unknown")
    except Exception:
        pass

    _safe_print("================================================================================")
    _safe_print("                     AI DOCUMENT CHATBOT - EVALUATION HARNESS                   ")
    _safe_print("================================================================================")
    _safe_print(f"API Target:       {args.api_url}")
    _safe_print(f"TOP_K Chunks:     {top_k_display}")
    _safe_print(f"Total Questions:  {len(all_questions)}")
    _safe_print(f"Verified to Run:  {len(questions_to_run)}")
    _safe_print(f"Sleep per query:  {args.sleep}s")
    _safe_print("--------------------------------------------------------------------------------")

    if not questions_to_run:
        _safe_print("\n[!] No verified questions found in questions.json.")
        _safe_print("    Every question is currently marked 'verified: false'.")
        _safe_print("    Please review expected pages and keywords, set 'verified: true' for verified items,")
        _safe_print("    or run with --all to test without filtering.")
        _safe_print("================================================================================")
        return 0

    # Locate test PDFs
    try:
        pdf_dir = find_test_pdfs_dir(args.pdf_dir)
        pdf_files = sorted(list(pdf_dir.glob("*.pdf")))
        _safe_print(f"Found {len(pdf_files)} PDF(s) in {pdf_dir}:")
        for p in pdf_files:
            _safe_print(f"  - {p.name} ({p.stat().st_size} bytes)")
    except Exception as exc:
        _safe_print(f"Error locating PDFs: {exc}")
        return 1

    # 1. Clear vector store
    _safe_print("\n[1/3] Clearing index via POST /clear...")
    try:
        clear_res = clear_index(args.api_url)
        _safe_print(f"      {clear_res.get('message', 'Index cleared.')}")
    except Exception as exc:
        _safe_print(f"Error clearing index: {exc}")
        _safe_print("Ensure the FastAPI backend is running (e.g. uvicorn app.main:app --reload --port 8000)")
        return 1

    # 2. Upload PDFs
    _safe_print("\n[2/3] Uploading and indexing PDFs via POST /upload...")
    for idx, pdf in enumerate(pdf_files, start=1):
        try:
            _safe_print(f"      [{idx}/{len(pdf_files)}] Uploading {pdf.name}...")
            res = upload_pdf(args.api_url, pdf)
            _safe_print(f"             Indexed {res.get('file_name')} ({res.get('pages')} pages, {res.get('chunks')} chunks)")
        except Exception as exc:
            _safe_print(f"Error uploading {pdf.name}: {exc}")
            return 1
        time.sleep(args.sleep)

    # 3. Run verified questions
    _safe_print(f"\n[3/3] Running {len(questions_to_run)} question(s) through POST /ask...")
    run_records = []
    response_times = []
    retrieval_times = []
    llm_times = []

    hit_count = 0
    refused_despite_hit_count = 0
    keyword_score_sum = 0.0
    kw_questions_count = 0
    ans_or_multi_count = 0

    refusal_count = 0
    unanswerable_count = 0

    _safe_print("\n" + "=" * 95)
    _safe_print(f"{'ID':<4} | {'TYPE':<12} | {'STATUS':<6} | {'TIME':<6} | {'DETAILS / REASON'}")
    _safe_print("-" * 95)

    for i, q in enumerate(questions_to_run, start=1):
        q_id = q["id"]
        q_text = q["question"]
        q_type = q.get("type", "answerable")
        exp_file = q.get("expected_file", "")
        exp_pages = set(q.get("expected_pages", []))
        exp_keywords = q.get("expected_keywords", [])

        try:
            resp = ask_question(args.api_url, q_text)
        except Exception as exc:
            _safe_print(f"{q_id:<4} | {q_type:<12} | FAIL   | {'--':<6} | Request failed: {exc}")
            run_records.append({
                "id": q_id,
                "question": q_text,
                "type": q_type,
                "status": "ERROR",
                "error": str(exc),
            })
            continue

        answer = resp.get("answer", "")
        sources = resp.get("sources", [])
        if top_k_display == "unknown" and resp.get("top_k"):
            top_k_display = resp.get("top_k")
        # Compute retrieval hit using "retrieved", not "sources"
        retrieved = resp.get("retrieved", [])
        rt = resp.get("response_time_seconds", 0.0)
        ret_t = resp.get("retrieval_seconds", 0.0)
        llm_t = resp.get("llm_seconds", 0.0)

        response_times.append(rt)
        retrieval_times.append(ret_t)
        llm_times.append(llm_t)

        passed = False
        reasons = []

        if q_type in ("answerable", "multi"):
            ans_or_multi_count += 1

            # 1. Retrieval hit computed from 'retrieved' chunks sent to LLM
            matched_retrieved = [
                r for r in retrieved
                if r.get("file_name") == exp_file and r.get("page_number") in exp_pages
            ]
            hit = len(matched_retrieved) > 0
            if hit:
                hit_count += 1
            else:
                retrieved_summary = [f"{r.get('file_name')} p.{r.get('page_number')}" for r in retrieved]
                reasons.append(f"Retrieval miss (expected {exp_file} p.{sorted(list(exp_pages))}; retrieved {retrieved_summary or 'none'})")

            # Check if model refused despite retrieval hit
            model_refused = is_refusal(answer)
            if hit and model_refused:
                refused_despite_hit_count += 1
                reasons.append("Refused despite retrieval hit")

            # 2. Keyword score computed only over non-empty expected_keywords
            if exp_keywords:
                kw_questions_count += 1
                found_keywords = [
                    kw for kw in exp_keywords
                    if kw.lower() in answer.lower()
                ]
                kw_score = len(found_keywords) / len(exp_keywords)
                keyword_score_sum += kw_score

                if kw_score < 1.0:
                    missing_kw = [kw for kw in exp_keywords if kw not in found_keywords]
                    reasons.append(f"Missing keywords: {missing_kw}")

                kw_passed = (kw_score >= 0.99)
            else:
                # An empty list must never count as a pass
                kw_passed = False
                reasons.append("Empty expected_keywords (cannot pass)")

            passed = (hit and kw_passed and not model_refused)
            details = "All expected retrieved chunks and keywords matched" if passed else "; ".join(reasons)

        elif q_type == "unanswerable":
            unanswerable_count += 1
            refused = is_refusal(answer)
            if refused:
                refusal_count += 1
                passed = True
                details = "Refusal confirmed (model admitted lack of context)"
            else:
                passed = False
                snippet = (answer[:80] + "...") if len(answer) > 80 else answer
                details = f"Failed refusal (answered: {snippet!r})"

        else:
            details = f"Unknown question type: {q_type}"

        status_str = "PASS" if passed else "FAIL"
        time_str = f"{rt:.2f}s"
        _safe_print(f"{q_id:<4} | {q_type:<12} | {status_str:<6} | {time_str:<6} | {details}")

        run_records.append({
            "id": q_id,
            "question": q_text,
            "type": q_type,
            "expected_file": exp_file,
            "expected_pages": list(exp_pages),
            "expected_keywords": exp_keywords,
            "actual_answer": answer,
            "actual_sources": sources,
            "actual_retrieved": retrieved,
            "timing": {
                "response_time_seconds": rt,
                "retrieval_seconds": ret_t,
                "llm_seconds": llm_t,
            },
            "passed": passed,
            "details": details,
        })

        # Throttle between queries
        if i < len(questions_to_run):
            time.sleep(args.sleep)

    _safe_print("=" * 95)

    # Compute summary metrics
    hit_rate = (hit_count / ans_or_multi_count) if ans_or_multi_count > 0 else 0.0
    if kw_questions_count > 0:
        avg_kw_score = keyword_score_sum / kw_questions_count
        kw_display = f"{avg_kw_score * 100:.1f}% ({kw_questions_count} question(s))"
    else:
        avg_kw_score = None
        kw_display = "n/a (0 questions)"

    refusal_rate = (refusal_count / unanswerable_count) if unanswerable_count > 0 else 0.0

    mean_rt = statistics.mean(response_times) if response_times else 0.0
    median_rt = statistics.median(response_times) if response_times else 0.0
    p95_rt = calculate_p95(response_times)
    mean_ret = statistics.mean(retrieval_times) if retrieval_times else 0.0
    mean_llm = statistics.mean(llm_times) if llm_times else 0.0

    _safe_print("\n================================================================================")
    _safe_print("                              EVALUATION SUMMARY                                ")
    _safe_print("================================================================================")
    _safe_print(f"TOP_K Chunks:                        {top_k_display}")
    _safe_print(f"Questions Evaluated:                 {len(questions_to_run)}")
    _safe_print(f"  - Answerable / Multi-hop:          {ans_or_multi_count}")
    _safe_print(f"  - Unanswerable:                    {unanswerable_count}")
    _safe_print("--------------------------------------------------------------------------------")
    _safe_print(f"Retrieval Hit Rate:                  {hit_rate * 100:.1f}% ({hit_count}/{ans_or_multi_count})")
    _safe_print(f"Refused Despite Retrieval Hit:       {refused_despite_hit_count}")
    _safe_print(f"Keyword Answer Score:                {kw_display}")
    _safe_print(f"Refusal Rate (Unanswerable):         {refusal_rate * 100:.1f}% ({refusal_count}/{unanswerable_count})")
    _safe_print("--------------------------------------------------------------------------------")
    _safe_print(f"Latency (Response Time):")
    _safe_print(f"  - Mean:                            {mean_rt:.2f}s")
    _safe_print(f"  - Median:                          {median_rt:.2f}s")
    _safe_print(f"  - P95:                             {p95_rt:.2f}s")
    _safe_print(f"  - Mean Retrieval Time:             {mean_ret:.2f}s")
    _safe_print(f"  - Mean LLM Generation Time:        {mean_llm:.2f}s")
    _safe_print("================================================================================")

    # Save results to results/results_<timestamp>.json
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = results_dir / f"results_{timestamp}.json"
    eval_payload = {
        "timestamp": timestamp,
        "top_k": top_k_display,
        "total_questions": len(all_questions),
        "evaluated_questions": len(questions_to_run),
        "metrics": {
            "retrieval_hit_rate": round(hit_rate, 4),
            "refused_despite_retrieval_hit_count": refused_despite_hit_count,
            "keyword_answer_score": round(avg_kw_score, 4) if avg_kw_score is not None else None,
            "keyword_questions_count": kw_questions_count,
            "refusal_rate": round(refusal_rate, 4),
            "latency": {
                "mean_response_time_seconds": round(mean_rt, 2),
                "median_response_time_seconds": round(median_rt, 2),
                "p95_response_time_seconds": round(p95_rt, 2),
                "mean_retrieval_seconds": round(mean_ret, 2),
                "mean_llm_seconds": round(mean_llm, 2),
            },
        },
        "records": run_records,
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(eval_payload, f, indent=2)

    _safe_print(f"\nRaw results successfully saved to: {out_file}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
