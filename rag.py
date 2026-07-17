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

from ingest import COLLECTION_NAME, DB_DIR, EMBEDDING_MODEL

load_dotenv()

MODEL = "claude-opus-4-8"

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


def retrieve(collection: chromadb.Collection, question: str, k: int = 3) -> list[dict]:
    """Return the k most relevant chunks as {text, source, chunk_index, distance}."""
    results = collection.query(query_texts=[question], n_results=k)
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


def build_context_block(hits: list[dict]) -> str:
    """Format retrieved chunks as numbered, source-labeled blocks Claude can cite by number
    (e.g. "[1]"), and a human can trace back to the exact source file afterward."""
    parts = []
    for i, hit in enumerate(hits, start=1):
        parts.append(f"[{i}] (source: {hit['source']})\n{hit['text']}")
    return "\n\n".join(parts)


def answer_question(client: anthropic.Anthropic, collection: chromadb.Collection, question: str) -> None:
    """Run one full RAG turn for a single question: retrieve chunks, ask Claude to
    answer strictly from them, then print the answer alongside what was retrieved
    so a human can verify the citations against the actual source chunks."""
    hits = retrieve(collection, question)
    context = build_context_block(hits)

    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}],
    )
    answer = next(b.text for b in response.content if b.type == "text")

    print(f"\n{answer}\n")
    print("Sources retrieved:")
    for i, hit in enumerate(hits, start=1):
        print(f"  [{i}] {hit['source']} (chunk {hit['chunk_index']}, distance={hit['distance']:.3f})")


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
