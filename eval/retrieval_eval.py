"""Measures retrieval quality against eval_set.json — no LLM call, so it's free to re-run.

For each question, checks whether a chunk from the expected source document
appears in the top-k retrieved chunks (hit@k). This isolates retrieval
quality from generation quality: it tells you whether the right document was
*found*, independent of whether Claude phrased a good answer from it.

Run:
    python -m eval.retrieval_eval
"""
import json
import pathlib
import sys

# Add the project root to the import path so `rag` resolves when this file
# runs as `python -m eval.retrieval_eval` from outside the eval/ directory.
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from rag import get_collection, retrieve  # noqa: E402

EVAL_SET_PATH = pathlib.Path(__file__).parent / "eval_set.json"


def run_eval(k: int = 3) -> None:
    """Run every question in eval_set.json through retrieval only (no Claude call),
    print a per-question PASS/FAIL, and report the overall hit@k rate."""
    eval_cases = json.loads(EVAL_SET_PATH.read_text(encoding="utf-8"))
    collection = get_collection()

    hits = 0
    for case in eval_cases:
        results = retrieve(collection, case["question"], k=k)
        retrieved_sources = [r["source"] for r in results]
        hit = case["expected_source"] in retrieved_sources
        hits += hit

        status = "PASS" if hit else "FAIL"
        print(f"[{status}] {case['question']}")
        print(f"       expected: {case['expected_source']}  |  retrieved: {retrieved_sources}")

    print(f"\nHit@{k}: {hits}/{len(eval_cases)} ({hits / len(eval_cases):.0%})")


if __name__ == "__main__":
    run_eval()
