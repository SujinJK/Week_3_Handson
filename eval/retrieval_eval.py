"""Measures retrieval quality against eval_set.json — no LLM call, so it's free to re-run.

For each question, checks whether a chunk from the expected source document
appears in the top-k retrieved chunks (hit@k). This isolates retrieval
quality from generation quality: it tells you whether the right document was
*found*, independent of whether Claude phrased a good answer from it.

Compares three retrieval strategies side by side: plain vector search,
vector+BM25 fused with Reciprocal Rank Fusion, and that fused pool reranked
with a cross-encoder -- the full pipeline rag.py actually uses.

Run:
    python -m eval.retrieval_eval
"""
import json
import pathlib
import sys

# Add the project root to the import path so `rag` resolves when this file
# runs as `python -m eval.retrieval_eval` from outside the eval/ directory.
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from rag import INITIAL_K, get_collection, hybrid_retrieve, rerank, retrieve  # noqa: E402

EVAL_SET_PATH = pathlib.Path(__file__).parent / "eval_set.json"


def run_eval(k: int = 3) -> None:
    """Run every question in eval_set.json through retrieval only (no Claude call),
    print a per-question PASS/FAIL for vector-only, hybrid, and reranked
    retrieval side by side, and report all three overall hit@k rates."""
    eval_cases = json.loads(EVAL_SET_PATH.read_text(encoding="utf-8"))
    collection = get_collection()

    vector_hits = 0
    hybrid_hits = 0
    reranked_hits = 0
    for case in eval_cases:
        question = case["question"]

        vector_candidates = retrieve(collection, question, k=INITIAL_K)
        vector_sources = [c["source"] for c in vector_candidates[:k]]
        vector_hit = case["expected_source"] in vector_sources
        vector_hits += vector_hit

        hybrid_candidates = hybrid_retrieve(collection, question, k=INITIAL_K)
        hybrid_sources = [c["source"] for c in hybrid_candidates[:k]]
        hybrid_hit = case["expected_source"] in hybrid_sources
        hybrid_hits += hybrid_hit

        reranked = rerank(question, list(hybrid_candidates), top_n=k)
        reranked_sources = [r["source"] for r in reranked]
        reranked_hit = case["expected_source"] in reranked_sources
        reranked_hits += reranked_hit

        status = "PASS" if vector_hit and hybrid_hit and reranked_hit else "FAIL"
        print(f"[{status}] {question}")
        print(f"       expected:         {case['expected_source']}")
        print(f"       vector-only top-{k}: {vector_sources}")
        print(f"       hybrid (RRF) top-{k}: {hybrid_sources}")
        print(f"       hybrid+reranked top-{k}: {reranked_sources}")

    total = len(eval_cases)
    print(f"\nHit@{k} (vector search only):              {vector_hits}/{total} ({vector_hits / total:.0%})")
    print(f"Hit@{k} (vector + BM25 fused, RRF):         {hybrid_hits}/{total} ({hybrid_hits / total:.0%})")
    print(f"Hit@{k} (fused, then reranked -- full pipeline): {reranked_hits}/{total} ({reranked_hits / total:.0%})")


if __name__ == "__main__":
    run_eval()
