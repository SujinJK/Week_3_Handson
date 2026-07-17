"""Ingest pipeline: load corpus files -> chunk -> embed locally -> store in Chroma.

Run this once (or whenever corpus/ changes) before running rag.py:
    python ingest.py
"""
import pathlib

import chromadb
from chromadb.utils import embedding_functions

from chunking import chunk_text

CORPUS_DIR = pathlib.Path(__file__).parent / "corpus"
DB_DIR = pathlib.Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "nimbus_docs"

# Runs entirely on-device, no API key or network calls needed.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def build_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(DB_DIR))
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    # Fresh collection every run so re-ingesting doesn't duplicate or leave stale chunks.
    try:
        client.delete_collection(COLLECTION_NAME)
    except (ValueError, chromadb.errors.NotFoundError):
        pass
    collection = client.create_collection(name=COLLECTION_NAME, embedding_function=embedding_fn)

    ids, documents, metadatas = [], [], []
    for path in sorted(CORPUS_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        for i, chunk in enumerate(chunk_text(text)):
            ids.append(f"{path.stem}::{i}")
            documents.append(chunk)
            metadatas.append({"source": path.name, "chunk_index": i})

    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    return collection


def main() -> None:
    collection = build_collection()
    print(f"Ingested {collection.count()} chunks from {CORPUS_DIR} into {DB_DIR}")


if __name__ == "__main__":
    main()
