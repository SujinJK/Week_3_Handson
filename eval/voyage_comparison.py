"""Compares local sentence-transformers embeddings against Voyage AI embeddings
on the same retrieval eval set, to see whether Anthropic's recommended
embeddings provider actually improves retrieval on this corpus.

Uses the Voyage client directly rather than Chroma's built-in
VoyageAIEmbeddingFunction, because a single Chroma collection is bound to one
embedding function used for both indexing and querying -- it can't apply
Voyage's different, retrieval-optimized input_type ("document" vs "query")
to the two sides. Embedding both sides ourselves and passing precomputed
vectors into an in-memory Chroma collection reproduces Voyage's recommended
usage correctly.

Costs a small amount of Voyage API credit -- comfortably inside the free
200M-token tier for a corpus this size (a few thousand tokens total).

Run:
    python -m eval.voyage_comparison
"""
import json
import pathlib
import sys

import chromadb
import voyageai
from dotenv import load_dotenv

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from chunking import semantic_chunk_text  # noqa: E402
from rag import get_collection, retrieve  # noqa: E402

load_dotenv()

CORPUS_DIR = pathlib.Path(__file__).parent.parent / "corpus"
EVAL_SET_PATH = pathlib.Path(__file__).parent / "eval_set.json"
VOYAGE_MODEL = "voyage-4"


def build_voyage_collection(client: voyageai.Client) -> chromadb.Collection:
    """Chunk the corpus the same way ingest.py does, embed with Voyage
    (input_type="document"), and store the vectors directly in an in-memory
    Chroma collection with no bound embedding function -- we supply
    embeddings ourselves for both indexing and querying."""
    chroma_client = chromadb.EphemeralClient()
    collection = chroma_client.create_collection(name="voyage_demo")

    ids, documents, metadatas = [], [], []
    for path in sorted(CORPUS_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        for i, chunk in enumerate(semantic_chunk_text(text)):
            ids.append(f"{path.stem}::{i}")
            documents.append(chunk)
            metadatas.append({"source": path.name, "chunk_index": i})

    result = client.embed(documents, model=VOYAGE_MODEL, input_type="document")
    collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=result.embeddings)
    return collection


def voyage_retrieve(collection: chromadb.Collection, query_embedding: list[float], k: int) -> list[str]:
    """Search the Voyage-embedded collection with an already-computed query vector."""
    hits = collection.query(query_embeddings=[query_embedding], n_results=k)
    return [meta["source"] for meta in hits["metadatas"][0]]


def run_comparison(k: int = 3) -> None:
    eval_cases = json.loads(EVAL_SET_PATH.read_text(encoding="utf-8"))
    voyage_client = voyageai.Client()

    print(f"Embedding corpus with Voyage AI ({VOYAGE_MODEL})...")
    voyage_collection = build_voyage_collection(voyage_client)
    local_collection = get_collection()

    # Batch every question into a single embed() call rather than one call
    # per question -- without a payment method on file, Voyage's free tier
    # is rate-limited to 3 requests/minute, and a per-question loop trips
    # that almost immediately. Batching is also just better practice.
    questions = [case["question"] for case in eval_cases]
    query_result = voyage_client.embed(questions, model=VOYAGE_MODEL, input_type="query")

    local_hits = 0
    voyage_hits = 0
    for case, query_embedding in zip(eval_cases, query_result.embeddings):
        local_sources = [r["source"] for r in retrieve(local_collection, case["question"], k=k)]
        voyage_sources = voyage_retrieve(voyage_collection, query_embedding, k)

        local_hit = case["expected_source"] in local_sources
        voyage_hit = case["expected_source"] in voyage_sources
        local_hits += local_hit
        voyage_hits += voyage_hit

        status = "PASS" if local_hit and voyage_hit else "FAIL"
        print(f"[{status}] {case['question']}")
        print(f"       expected:               {case['expected_source']}")
        print(f"       local (MiniLM) top-{k}:    {local_sources}")
        print(f"       Voyage ({VOYAGE_MODEL}) top-{k}: {voyage_sources}")

    total = len(eval_cases)
    print(f"\nHit@{k} (local sentence-transformers, all-MiniLM-L6-v2): {local_hits}/{total} ({local_hits / total:.0%})")
    print(f"Hit@{k} (Voyage AI, {VOYAGE_MODEL}):                        {voyage_hits}/{total} ({voyage_hits / total:.0%})")


if __name__ == "__main__":
    run_comparison()
