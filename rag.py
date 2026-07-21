"""RAG Q&A over the Nimbus Cloud Storage sample corpus.

Retrieval is local and free (Chroma + sentence-transformers). Generation
calls Claude, grounded strictly in the retrieved chunks, with citations.

Run:
    python ingest.py   # once, to build the vector store
    python rag.py       # interactive Q&A loop
"""
import pathlib

import anthropic
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder

from ingest import COLLECTION_NAME, DB_DIR, EMBEDDING_MODEL

load_dotenv()

MODEL = "claude-opus-4-8"

# Runs entirely on-device, same as the embedding model -- reranking stays free.
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
INITIAL_K = 8  # wider candidate pool handed to the reranker
FINAL_K = 3    # how many chunks actually reach Claude, after reranking

_reranker: CrossEncoder | None = None

SYSTEM_PROMPT = (
    "You are a support assistant for Nimbus Cloud Storage. Answer the "
    "question using ONLY the numbered context snippets provided — do not "
    "use outside knowledge. Every claim in your answer must cite the "
    "snippet(s) it came from, like this: 'PTO caps at 21 days per year [1].' "
    "If the context snippets do not contain enough information to answer, "
    "say exactly: 'I don't have enough information in the provided "
    "documents to answer that.' Do not guess or fill gaps with assumptions."
)


def get_collection() -> chromadb.Collection:
    """Open the Chroma collection built by ingest.py. Must use the same embedding
    model as ingestion, or query vectors won't be comparable to the stored ones."""
    client = chromadb.PersistentClient(path=str(DB_DIR))
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    return client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_fn)


def retrieve(
    collection: chromadb.Collection, question: str, k: int = 3, where: dict | None = None
) -> list[dict]:
    """Return the k most relevant chunks as {text, source, chunk_index, distance}.

    `where` is a Chroma metadata filter (e.g. {"status": "current"}) applied
    BEFORE similarity ranking -- chunks that don't match are never candidates,
    not just ranked lower. Passing None searches the whole collection."""
    results = collection.query(query_texts=[question], n_results=k, where=where)
    hits = []
    for text, meta, distance in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        hits.append({
            "text": text,
            "source": meta["source"],
            "chunk_index": meta["chunk_index"],
            "distance": distance,
        })
    return hits


def get_reranker() -> CrossEncoder:
    """Lazily load the local cross-encoder reranker -- only needed once per process."""
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


def rerank(question: str, hits: list[dict], top_n: int = FINAL_K) -> list[dict]:
    """Re-score retrieved hits with a cross-encoder and keep the top_n by that score.

    Vector search (retrieve()) compares independently-computed embeddings --
    fast, but each chunk's vector is fixed before it ever "sees" the
    question. A cross-encoder reads the question and each chunk together and
    produces a more accurate relevance score -- too slow to run over an
    entire corpus, but cheap over the small candidate pool retrieve() already
    narrowed things down to. Mutates each hit in place, adding "rerank_score".
    """
    reranker = get_reranker()
    pairs = [(question, hit["text"]) for hit in hits]
    scores = reranker.predict(pairs)
    for hit, score in zip(hits, scores):
        hit["rerank_score"] = float(score)
    return sorted(hits, key=lambda h: h["rerank_score"], reverse=True)[:top_n]


def build_context_block(hits: list[dict]) -> str:
    """Format retrieved chunks as numbered, source-labeled blocks Claude can cite by number
    (e.g. "[1]"), and a human can trace back to the exact source file afterward."""
    parts = []
    for i, hit in enumerate(hits, start=1):
        parts.append(f"[{i}] (source: {hit['source']})\n{hit['text']}")
    return "\n\n".join(parts)


def generate_answer(client: anthropic.Anthropic, hits: list[dict], question: str) -> str:
    """Ask Claude to answer strictly from the given retrieved chunks. Returns the answer text.

    Split out from answer_question() so eval scripts can generate a real
    answer through the same code path the interactive app uses, instead of
    duplicating the Claude call."""
    context = build_context_block(hits)
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}],
    )
    return next(b.text for b in response.content if b.type == "text")


def answer_question(client: anthropic.Anthropic, collection: chromadb.Collection, question: str) -> None:
    """Run one full RAG turn for a single question: retrieve a wide candidate pool,
    rerank it down to the final few chunks, ask Claude to answer strictly from
    them, then print the answer alongside both retrieval stages so a human can
    see what the reranker kept vs. discarded and verify citations against the
    actual source chunks."""
    # Filters retrieval down to current documents only. Every real chunk is
    # tagged "current" (see ingest.py), so this has no effect today -- it's
    # here so a future superseded document is excluded by construction rather
    # than relying on generation to notice the conflict after the fact.
    candidates = retrieve(collection, question, k=INITIAL_K, where={"status": "current"})
    hits = rerank(question, candidates, top_n=FINAL_K)
    answer = generate_answer(client, hits, question)

    kept = {(h["source"], h["chunk_index"]) for h in hits}

    print(f"\n{answer}\n")
    print(f"Initial retrieval (top {len(candidates)} by vector distance; * = kept after rerank):")
    for hit in sorted(candidates, key=lambda h: h["distance"]):
        marker = "*" if (hit["source"], hit["chunk_index"]) in kept else " "
        print(
            f"  [{marker}] {hit['source']} (chunk {hit['chunk_index']}, "
            f"distance={hit['distance']:.3f}, rerank_score={hit['rerank_score']:.3f})"
        )
    print(f"\nFinal {len(hits)} sent to Claude, in rerank order:")
    for i, hit in enumerate(hits, start=1):
        print(f"  [{i}] {hit['source']} (chunk {hit['chunk_index']}, rerank_score={hit['rerank_score']:.3f})")


def main() -> None:
    """Entry point for `python rag.py` — the interactive question loop."""
    if not pathlib.Path(DB_DIR).exists():
        print("No vector store found. Run `python ingest.py` first.")
        return

    client = anthropic.Anthropic()
    collection = get_collection()

    print("Nimbus RAG Q&A. Ask a question, or /quit to exit.\n")
    while True:
        question = input("you> ").strip()
        if not question:
            continue
        if question in ("/quit", "/exit"):
            break
        answer_question(client, collection, question)


if __name__ == "__main__":
    main()
