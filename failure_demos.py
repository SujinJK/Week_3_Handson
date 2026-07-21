"""Deliberately breaks the RAG pipeline in three realistic ways, live against
the real API, so the failure modes described in README.md are something you
can actually see happen instead of just reading about.

Each demo is self-contained and uses its own temporary Chroma collection —
none of them touch chroma_db/ (the real pipeline's store built by ingest.py).

Run:
    python failure_demos.py
"""
import shutil
import tempfile
from pathlib import Path
from typing import Callable

import anthropic
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

from chunking import chunk_text, semantic_chunk_text
from ingest import EMBEDDING_MODEL
from rag import MODEL, SYSTEM_PROMPT, get_collection

load_dotenv()

CORPUS_DIR = Path(__file__).parent / "corpus"
CONFLICT_CORPUS_DIR = Path(__file__).parent / "corpus_conflict_demo"


def _build_temp_collection(
    corpus_dir: Path,
    chunk_fn: Callable[[str], list[str]],
    status_map: dict[str, str] | None = None,
) -> tuple[chromadb.Collection, str]:
    """Ingest a corpus into a fresh, disposable Chroma collection using the
    given chunking function. Returns (collection, temp_dir) — caller must
    clean up temp_dir.

    chunk_fn takes document text and returns a list of chunk strings — pass
    e.g. `lambda text: semantic_chunk_text(text, max_chunk_size=120)` or
    `lambda text: chunk_text(text, chunk_size=8, overlap=0)` to compare
    strategies. status_map optionally overrides the "status" metadata for
    specific filenames (e.g. {"employee_handbook_2023.md": "superseded"});
    any file not in the map defaults to "current", matching ingest.py."""
    status_map = status_map or {}
    temp_dir = tempfile.mkdtemp(prefix="rag_failure_demo_")
    client = chromadb.PersistentClient(path=temp_dir)
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    collection = client.create_collection(name="demo", embedding_function=embedding_fn)

    ids, documents, metadatas = [], [], []
    for path in sorted(corpus_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        status = status_map.get(path.name, "current")
        for i, chunk in enumerate(chunk_fn(text)):
            ids.append(f"{path.stem}::{i}")
            documents.append(chunk)
            metadatas.append({"source": path.name, "chunk_index": i, "status": status})
    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    return collection, temp_dir


def _ask(client: anthropic.Anthropic, system_prompt: str, context: str, question: str) -> str:
    response = client.messages.create(
        model=MODEL,
        max_tokens=512,
        system=system_prompt,
        messages=[{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}],
    )
    return next(b.text for b in response.content if b.type == "text")


def demo_1_grounding_failure(client: anthropic.Anthropic) -> None:
    """Failure mode: hallucination / extrapolation beyond the retrieved context.

    The retrieved chunk is about EMPLOYEE MFA for production system access.
    The question asks about CUSTOMER-facing login 2FA — a related but
    different, and never-documented, topic. A weakly grounded system prompt
    tends to blend the two and answer as if customer 2FA were documented.
    """
    print("=" * 70)
    print("DEMO 1: Grounding failure (hallucination by extrapolation)")
    print("=" * 70)

    # Retrieval-side blind spot: nearest-neighbor search has no "nothing is
    # actually relevant" signal — it always returns k results, just some
    # results are closer than others. Ask something the corpus doesn't cover
    # at all, and it still comes back with "matches."
    off_topic_question = "What is Nimbus's uptime SLA guarantee percentage?"
    real_collection = get_collection()
    off_topic_hits = real_collection.query(query_texts=[off_topic_question], n_results=3)
    print(f"\n--- Retrieval blind spot: no chunk in the corpus covers this topic ---")
    print(f"Question: {off_topic_question}")
    print("Retrieval still returns its top 3 'closest' chunks anyway:")
    for source, distance in zip(off_topic_hits["metadatas"][0], off_topic_hits["distances"][0]):
        print(f"  {source['source']} (distance={distance:.3f})")
    print(
        "None of these chunks actually answer the question — retrieval has "
        "no concept of 'not relevant enough,' only 'closest available.' "
        "Whether a bad answer gets through from here depends entirely on the "
        "generation step below refusing to use context that doesn't fit."
    )

    # Real chunk from security_policy.md — about employee MFA, not customer-facing 2FA.
    context = (
        "[1] (source: security_policy.md)\n"
        "MFA is mandatory for all accounts with access to production systems or "
        "customer data. Approved MFA methods are: hardware security keys "
        "(preferred), authenticator apps (TOTP), or SMS (only when the first two "
        "are unavailable)."
    )
    question = "Can Nimbus customers enable two-factor authentication on their own account login?"

    weak_prompt = "You are a helpful support assistant for Nimbus Cloud Storage. Answer the user's question."
    print(f"\n--- Weak system prompt (no grounding instruction) ---\n{weak_prompt}\n")
    print(_ask(client, weak_prompt, context, question))

    print(f"\n--- Our actual grounded system prompt ---\n")
    print(_ask(client, SYSTEM_PROMPT, context, question))

    print(
        "\n[Why this matters] The context is about an internal employee "
        "policy, not a customer-facing feature — a scope mismatch a weak "
        "prompt is prone to paper over. On a strong, well-calibrated model "
        "like Claude Opus, don't be surprised if BOTH prompts refuse here — "
        "that's a genuinely good result, not a broken demo. It doesn't mean "
        "the risk is fake: retrieval (above) proved it will hand generation "
        "irrelevant context without warning, on weaker models or trickier "
        "phrasing that same irrelevant context is exactly what gets blended "
        "into a confident, wrong answer. The system prompt's explicit "
        "'ONLY use context' + 'say so if you can't answer' instruction is "
        "the deliberate safety net for that — never skip it on the assumption "
        "the model will 'just know' not to guess.\n"
    )


def demo_2_chunking_failure(client: anthropic.Anthropic) -> None:
    """Failure mode: bad chunking fragments the fact the question needs.

    refund_policy.md has one sentence containing both "full refund" and
    "5 business days". With very small, non-overlapping word-count chunks,
    that sentence gets split across chunk boundaries, so no single retrieved
    chunk contains the complete fact. Semantic chunking (what ingest.py
    actually uses) never splits a sentence, at any target size -- shown
    here at an aggressively small max_chunk_size to prove the point.
    """
    print("=" * 70)
    print("DEMO 2: Chunking failure (answer fragmented across chunk boundaries)")
    print("=" * 70)

    question = "If I'm charged incorrectly, how many business days until I get a refund?"

    # Normal pipeline chunking (what ingest.py actually uses) — control case.
    good_collection, good_dir = _build_temp_collection(
        CORPUS_DIR, chunk_fn=lambda text: semantic_chunk_text(text, max_chunk_size=120)
    )
    good_hits = good_collection.query(query_texts=[question], n_results=1)
    good_context = good_hits["documents"][0][0]
    print(f"\n--- Semantic chunking, 120-word target (normal) — retrieved chunk ---\n{good_context}\n")
    print(_ask(client, SYSTEM_PROMPT, f"[1] (source: refund_policy.md)\n{good_context}", question))

    # Deliberately broken: tiny word-count chunks, no overlap.
    bad_collection, bad_dir = _build_temp_collection(
        CORPUS_DIR, chunk_fn=lambda text: chunk_text(text, chunk_size=8, overlap=0)
    )
    bad_hits = bad_collection.query(query_texts=[question], n_results=1)
    bad_context = bad_hits["documents"][0][0]
    print(f"\n--- Word-count chunking, 8 words, no overlap (broken) — retrieved chunk ---\n{bad_context}\n")
    print(_ask(client, SYSTEM_PROMPT, f"[1] (source: refund_policy.md)\n{bad_context}", question))

    # Semantic chunking pushed to an aggressively small target size (15
    # words) -- the key sentence is ~30 words, far over target, but it still
    # comes back whole as its own over-sized chunk rather than being cut.
    tiny_collection, tiny_dir = _build_temp_collection(
        CORPUS_DIR, chunk_fn=lambda text: semantic_chunk_text(text, max_chunk_size=15)
    )
    tiny_hits = tiny_collection.query(query_texts=[question], n_results=1)
    tiny_context = tiny_hits["documents"][0][0]
    print(f"\n--- Semantic chunking, 15-word target (small on purpose) — retrieved chunk ---\n{tiny_context}\n")
    print(_ask(client, SYSTEM_PROMPT, f"[1] (source: refund_policy.md)\n{tiny_context}", question))

    shutil.rmtree(good_dir, ignore_errors=True)
    shutil.rmtree(bad_dir, ignore_errors=True)
    shutil.rmtree(tiny_dir, ignore_errors=True)

    print(
        "\n[Why this matters] The fact ('5 business days') and the claim it "
        "attaches to ('full refund') live in the same sentence. Word-count "
        "chunking splits them apart whenever the target size happens to cut "
        "through that sentence — the grounded system prompt then has to "
        "either guess or correctly refuse, neither as good as just answering "
        "correctly. Semantic chunking can't make this specific mistake: it "
        "only ever breaks between sentences, so even at a target size far "
        "smaller than the sentence itself, the sentence comes back whole "
        "(as an over-sized chunk) rather than fragmented. The tradeoff is a "
        "less predictable chunk size, not a wrong answer.\n"
    )


def demo_3_conflicting_sources_failure(client: anthropic.Anthropic) -> None:
    """Failure mode: a stale/superseded document conflicts with the current one.

    corpus_conflict_demo/ contains both the current employee_handbook.md
    (5-day PTO carryover cap) and an old employee_handbook_2023.md (10-day
    cap) — nothing marks either one as outdated, which is realistic: stale
    documents rarely announce themselves.
    """
    print("=" * 70)
    print("DEMO 3: Conflicting sources (stale document confuses retrieval)")
    print("=" * 70)

    question = "How many days of PTO carry over into the next year?"

    # File content alone doesn't mark either handbook as outdated -- that's
    # realistic. The status tag below simulates knowing that out-of-band
    # (e.g. from a document management system), which is exactly the piece
    # metadata filtering needs in order to do anything.
    status_map = {"employee_handbook_2023.md": "superseded"}
    collection, temp_dir = _build_temp_collection(
        CONFLICT_CORPUS_DIR,
        chunk_fn=lambda text: semantic_chunk_text(text, max_chunk_size=120),
        status_map=status_map,
    )

    # k=1, NO filter: if the stale doc ranks closer than the current one,
    # this silently returns the WRONG policy with full confidence — no
    # conflict to flag, because the conflicting chunk was never retrieved.
    top1 = collection.query(query_texts=[question], n_results=1)
    top1_source = top1["metadatas"][0][0]["source"]
    top1_context = f"[1] (source: {top1_source})\n{top1['documents'][0][0]}"
    print(f"\n--- k=1, no filter — retrieved: {top1_source} ---")
    print(_ask(client, SYSTEM_PROMPT, top1_context, question))

    # k=3, NO filter: both versions likely appear, so the model can at least
    # see the conflict.
    hits = collection.query(query_texts=[question], n_results=3)
    context_hits = [
        {"text": text, "source": meta["source"]}
        for text, meta in zip(hits["documents"][0], hits["metadatas"][0])
    ]
    context = "\n\n".join(
        f"[{i+1}] (source: {h['source']})\n{h['text']}" for i, h in enumerate(context_hits)
    )

    print("\n--- k=3, no filter — both the current AND the stale handbook get retrieved ---")
    for i, h in enumerate(context_hits, start=1):
        print(f"  [{i}] {h['source']}")
    print(_ask(client, SYSTEM_PROMPT, context, question))

    # k=1, WITH metadata filtering (where={"status": "current"}): the stale
    # chunk is excluded from the candidate pool before ranking even happens
    # -- not outranked, not flagged, just never in the running.
    filtered_top1 = collection.query(query_texts=[question], n_results=1, where={"status": "current"})
    filtered_source = filtered_top1["metadatas"][0][0]["source"]
    filtered_context = f"[1] (source: {filtered_source})\n{filtered_top1['documents'][0][0]}"
    print(f"\n--- k=1, WITH filter (status=current) — retrieved: {filtered_source} ---")
    print(_ask(client, SYSTEM_PROMPT, filtered_context, question))

    shutil.rmtree(temp_dir, ignore_errors=True)

    print(
        "\n[Why this matters] Nothing in the file content marks "
        "employee_handbook_2023.md as outdated — a plausible real-world "
        "accident (an old doc never removed from a shared drive). At k=1 with "
        "no filter, if the stale chunk happens to rank closer, the system "
        "returns a confident, cleanly-cited, WRONG answer — the worst kind of "
        "failure, because nothing about the response looks uncertain. At k=3 "
        "the model can at least see both versions and flag the conflict "
        "instead of picking one — better, but still relies on generation to "
        "notice what retrieval should never have surfaced unresolved. "
        "Metadata filtering fixes it at the source: tagging the stale "
        "document and excluding it with a `where` clause means it's never a "
        "retrieval candidate at all, at any k — no reliance on generation "
        "noticing anything. The catch is that filter is only as good as the "
        "status tag behind it: something still has to mark a document "
        "superseded in the first place, which is a document lifecycle "
        "problem, not a retrieval-code problem.\n"
    )


def main() -> None:
    client = anthropic.Anthropic()
    demo_1_grounding_failure(client)
    demo_2_chunking_failure(client)
    demo_3_conflicting_sources_failure(client)


if __name__ == "__main__":
    main()
