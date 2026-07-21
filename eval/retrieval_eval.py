"""Measures retrieval quality against eval_set.json — no LLM call, so it's free to re-run.

For each question, checks whether a chunk from the expected source document
appears in the top-k retrieved chunks (hit@k). This isolates retrieval
quality from generation quality: it tells you whether the right document was
*found*, independent of whether Claude phrased a good answer from it.

Also compares plain vector-search hit@k against hit@k after reranking, to
show directly whether the cross-encoder changes anything on this corpus.

Run:
    python -m eval.retrieval_eval
"""
import json
import pathlib
import sys

# Add the project root to the import path so `rag` resolves when this file
# runs as `python -m eval.retrieval_eval` from outside the eval/ directory.
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from rag import INITIAL_K, get_collection, rerank, retrieve  # noqa: E402

EVAL_SET_PATH = pathlib.Path(__file__).parent / "eval_set.json"


def run_eval(k: int = 3) -> None:
    """Run every question in eval_set.json through retrieval only (no Claude call),
    print a per-question PASS/FAIL for plain vector search and for reranked
    retrieval side by side, and report both overall hit@k rates."""
    eval_cases = json.loads(EVAL_SET_PATH.read_text(encoding="utf-8"))
    collection = get_collection()

    plain_hits = 0
    reranked_hits = 0
    for case in eval_cases:
        candidates = retrieve(collection, case["question"], k=INITIAL_K)

        plain_sources = [c["source"] for c in candidates[:k]]
        plain_hit = case["expected_source"] in plain_sources
        plain_hits += plain_hit

        reranked = rerank(case["question"], list(candidates), top_n=k)
        reranked_sources = [r["source"] for r in reranked]
        reranked_hit = case["expected_source"] in reranked_sources
        reranked_hits += reranked_hit

        status = "PASS" if plain_hit and reranked_hit else "FAIL"
        print(f"[{status}] {case['question']}")
        print(f"       expected: {case['expected_source']}")
        print(f"       vector-only top-{k}:  {plain_sources}")
        print(f"       reranked top-{k}:     {reranked_sources}")

    total = len(eval_cases)
    print(f"\nHit@{k} (vector search only):        {plain_hits}/{total} ({plain_hits / total:.0%})")
    print(f"Hit@{k} (retrieve {INITIAL_K}, then rerank to {k}): {reranked_hits}/{total} ({reranked_hits / total:.0%})")


if __name__ == "__main__":
    run_eval()
