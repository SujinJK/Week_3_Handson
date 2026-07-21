"""RAG Q&A over the Nimbus Cloud Storage sample corpus.

Retrieval is local and free (Chroma + sentence-transformers). Generation
calls Claude, grounded strictly in the retrieved chunks, with citations.

Run:
    python ingest.py   # once, to build the vector store
    python rag.py       # interactive Q&A loop
"""
import pathlib
import re

import anthropic
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from ingest import COLLECTION_NAME, DB_DIR, EMBEDDING_MODEL

load_dotenv()

MODEL = "claude-opus-4-8"

# Runs entirely on-device, same as the embedding model -- reranking stays free.
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
INITIAL_K = 8  # wider candidate pool handed to the reranker
FINAL_K = 3    # how many chunks actually reach Claude, after reranking
RRF_K = 60     # standard damping constant for Reciprocal Rank Fusion

_reranker: CrossEncoder | None = None
_bm25_index_cache: dict[int, tuple[BM25Okapi, list[dict]]] = {}
_TOKEN_RE = re.compile(r"\w+")

HYDE_SYSTEM_PROMPT = (
    "Write a single short passage (1-2 sentences) that would plausibly "
    "appear in Nimbus Cloud Storage's internal policy documents and would "
    "directly answer the user's question. State a plausible-sounding policy "
    "as flat fact, in the same neutral, factual style as a company "
    "handbook or FAQ — do not hedge, do not say you're unsure, do not "
    "address the user directly. This hypothetical passage is only used to "
    "improve document search; it is never shown to the user."
)

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


def _tokenize(text: str) -> list[str]:
    """Lowercase word tokenizer for BM25 -- doesn't need to match embedding
    preprocessing, just needs to be consistent between corpus and query."""
    return _TOKEN_RE.findall(text.lower())


def _get_bm25_index(collection: chromadb.Collection) -> tuple[BM25Okapi, list[dict]]:
    """Build (and cache, keyed by collection object identity) a BM25 index over
    every current chunk in the collection."""
    cache_key = id(collection)
    if cache_key not in _bm25_index_cache:
        result = collection.get(where={"status": "current"})
        chunks = [
            {"text": text, "source": meta["source"], "chunk_index": meta["chunk_index"]}
            for text, meta in zip(result["documents"], result["metadatas"])
        ]
        tokenized_corpus = [_tokenize(c["text"]) for c in chunks]
        _bm25_index_cache[cache_key] = (BM25Okapi(tokenized_corpus), chunks)
    return _bm25_index_cache[cache_key]


def bm25_search(collection: chromadb.Collection, question: str, k: int) -> list[dict]:
    """Keyword search over the corpus using BM25 -- exact term overlap, unlike
    vector search's semantic similarity. Catches proper nouns, IDs, and
    specific terminology that an embedding model can under-weight."""
    bm25, chunks = _get_bm25_index(collection)
    scores = bm25.get_scores(_tokenize(question))
    ranked = sorted(zip(chunks, scores), key=lambda pair: pair[1], reverse=True)
    return [{**chunk, "bm25_score": float(score)} for chunk, score in ranked[:k]]


def hybrid_retrieve(collection: chromadb.Collection, question: str, k: int = INITIAL_K) -> list[dict]:
    """Fuse vector search and BM25 keyword search with Reciprocal Rank Fusion
    (RRF), so a chunk that ranks well on EITHER signal surfaces as a
    candidate -- not just chunks that win on semantic similarity alone.

    RRF scores each chunk by 1/(RRF_K + rank) in each ranked list it appears
    in, then sums across lists. It only needs rank position, not raw score
    magnitude, which sidesteps the fact that vector distance and BM25 score
    aren't on comparable scales.
    """
    vector_hits = retrieve(collection, question, k=k, where={"status": "current"})
    keyword_hits = bm25_search(collection, question, k=k)

    fused_scores: dict[tuple, float] = {}
    chunk_lookup: dict[tuple, dict] = {}
    for ranked_list in (vector_hits, keyword_hits):
        for rank, hit in enumerate(ranked_list):
            key = (hit["source"], hit["chunk_index"])
            fused_scores[key] = fused_scores.get(key, 0.0) + 1.0 / (RRF_K + rank + 1)
            chunk_lookup[key] = {**chunk_lookup.get(key, {}), **hit}

    ranked_keys = sorted(fused_scores, key=lambda key: fused_scores[key], reverse=True)
    results = []
    for key in ranked_keys[:k]:
        hit = dict(chunk_lookup[key])
        hit["rrf_score"] = fused_scores[key]
        results.append(hit)
    return results


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


def hyde_rewrite(client: anthropic.Anthropic, question: str) -> str:
    """HyDE (Hypothetical Document Embeddings): ask Claude to imagine what a
    real answer to this question would look like, written in the corpus's
    own style, then embed THAT instead of the raw question.

    Bridges the gap between a casual question and formal document phrasing --
    a colloquial question and a policy sentence can mean the same thing while
    barely sharing any words, which hurts vector similarity even though the
    embedding model handles paraphrasing well in general. Costs one extra
    Claude call per question, so unlike every other retrieval technique in
    this file, it is NOT free -- not wired into answer_question() by default
    for that reason. See eval/query_transformation_comparison.py.
    """
    response = client.messages.create(
        model=MODEL,
        max_tokens=150,
        system=HYDE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": question}],
    )
    return next(b.text for b in response.content if b.type == "text")


def answer_question(client: anthropic.Anthropic, collection: chromadb.Collection, question: str) -> None:
    """Run one full RAG turn for a single question: fuse vector + keyword search
    into a wide candidate pool, rerank it down to the final few chunks, ask
    Claude to answer strictly from them, then print the answer alongside both
    retrieval stages so a human can see what the reranker kept vs. discarded
    and verify citations against the actual source chunks."""
    # hybrid_retrieve() filters to current documents only (via retrieve()'s
    # where clause) and fuses it with BM25 keyword search. Every real chunk
    # is tagged "current" (see ingest.py), so the status filter is a no-op
    # today -- it's here so a future superseded document is excluded by
    # construction rather than relying on generation to notice the conflict.
    candidates = hybrid_retrieve(collection, question, k=INITIAL_K)
    hits = rerank(question, candidates, top_n=FINAL_K)
    answer = generate_answer(client, hits, question)

    kept = {(h["source"], h["chunk_index"]) for h in hits}

    print(f"\n{answer}\n")
    print(f"Initial retrieval (top {len(candidates)} by vector+keyword fusion; * = kept after rerank):")
    for hit in sorted(candidates, key=lambda h: h["rrf_score"], reverse=True):
        marker = "*" if (hit["source"], hit["chunk_index"]) in kept else " "
        signals = []
        if "distance" in hit:
            signals.append(f"distance={hit['distance']:.3f}")
        if "bm25_score" in hit:
            signals.append(f"bm25={hit['bm25_score']:.2f}")
        print(
            f"  [{marker}] {hit['source']} (chunk {hit['chunk_index']}, "
            f"rrf={hit['rrf_score']:.4f}, {', '.join(signals)}, "
            f"rerank_score={hit['rerank_score']:.3f})"
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
