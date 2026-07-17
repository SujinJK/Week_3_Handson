# Week 3 Handson

Retrieval-Augmented Generation (RAG) — a Q&A app that answers questions about
a small fictional company ("Nimbus Cloud Storage") strictly from its own
internal documents, citing sources, instead of relying on Claude's general
knowledge.

## Pipeline

```
corpus/*.md --chunk--> chunking.py --embed (local)--> ingest.py --store--> Chroma
                                                                              |
                                                                              v
                                          question --embed (local)--> retrieve top-k
                                                                              |
                                                                              v
                                                          Claude (generate, cite sources)
```

Retrieval (chunking, embedding, vector search) is entirely local and free —
no API calls, no cost. Only the final answer-generation step calls Claude.

## Project structure

| File | Purpose |
|---|---|
| `corpus/*.md` | 5 sample "company knowledge base" documents (HR, security, product, billing, incidents) — deliberately on different topics so retrieval quality is testable. |
| `chunking.py` | Splits document text into overlapping word-count chunks. Pure Python, no dependencies — easy to unit test. |
| `ingest.py` | Loads `corpus/`, chunks each file, embeds chunks with a local model, stores them in a persistent Chroma collection (`chroma_db/`). |
| `rag.py` | The Q&A app: embeds the question, retrieves the top-k most similar chunks, sends them to Claude as context, prints the cited answer and which chunks were retrieved. |
| `eval/eval_set.json` | 8 question -> expected-source pairs, used to measure retrieval quality. |
| `eval/retrieval_eval.py` | Runs every question in the eval set through retrieval only (no Claude call, so it's free) and reports hit@k — did the right document get retrieved? |
| `tests/test_chunking.py` | Unit tests for the chunking logic (chunk boundaries, overlap, no data loss). |
| `requirements.txt` / `requirements-dev.txt` | Runtime deps (`anthropic`, `chromadb`, `sentence-transformers`) / test deps (`pytest`). |
| `.env.example` / `.env` | API key template / your real key (gitignored). |

## Setup

```
pip install -r requirements.txt
cp .env.example .env   # then add your ANTHROPIC_API_KEY
```

**Windows note:** if `import torch` fails with `WinError 1114` (a DLL
initialization error), your Visual C++ Redistributable is out of date —
install the latest from
[aka.ms/vs/17/release/vc_redist.x64.exe](https://aka.ms/vs/17/release/vc_redist.x64.exe).
This is unrelated to the Python packages themselves; PyTorch's compiled
extensions need a current C++ runtime.

## Running it

```
python ingest.py          # build the vector store (run once, or whenever corpus/ changes)
python rag.py              # interactive Q&A loop
python -m eval.retrieval_eval   # measure retrieval quality (free, no API calls)
pytest                     # unit tests for chunking
```

## Embeddings and vector similarity

Each chunk of text is converted into a vector (384 numbers, from the
`all-MiniLM-L6-v2` model) that captures its meaning — chunks about similar
topics end up as vectors that point in similar directions. A question is
embedded the same way, and Chroma finds the stored chunks whose vectors are
**closest** to the question's vector (cosine distance — lower means more
similar). That's the entire "search" step: no keyword matching, so a
question phrased differently from the source text can still retrieve it.

**What we tuned and why it matters:**
- **Chunk size (120 words) and overlap (30 words)** in `chunking.py` — too
  large and irrelevant text dilutes the embedding, drowning out the specific
  fact being asked about; too small and a chunk loses the surrounding
  sentence that explains it. Overlap prevents an answer from being split
  exactly at a chunk boundary.
- **k=3** retrieved chunks per question — enough that the right chunk is
  very likely included even if it's not the single closest match, without
  flooding Claude's context with irrelevant chunks that could get cited by
  mistake.

## How RAG fails (and how this project detects each)

1. **Retrieval miss** — the right document exists but doesn't get retrieved
   (wrong chunk size, bad phrasing, embedding model doesn't capture the
   similarity). *Detected by:* `eval/retrieval_eval.py` — it isolates
   retrieval from generation and reports hit@k per question, so a drop here
   is unambiguously a retrieval problem, not a wording problem in Claude's
   answer.
2. **Hallucination / answering from outside the context** — the model
   answers confidently using its own general knowledge instead of admitting
   the documents don't cover it. *Detected by:* the "CEO's favorite
   programming language" test case in `rag.py` — a question with no possible
   answer in the corpus. The system prompt forces an explicit "I don't have
   enough information" refusal instead of a guess; if that refusal stops
   happening, generation is hallucinating.
3. **Wrong or missing citations** — the answer is correct but doesn't cite
   the chunk it came from, or cites the wrong one, so a human can't verify
   it. *Detected by:* manually spot-checking that each citation number in
   the printed answer matches a source that actually supports that claim —
   automatable by asking a second Claude call to verify each cited claim
   against its source chunk (not implemented here, but the natural next
   step if this were a production system).

## Test cases

**`tests/test_chunking.py`**
- `test_empty_string_returns_no_chunks` — empty input produces no chunks
- `test_short_text_returns_single_chunk` — text shorter than chunk_size isn't split
- `test_splits_into_multiple_chunks_when_over_size` — long text is split
- `test_consecutive_chunks_overlap` — the overlap words actually repeat between chunks
- `test_all_words_are_preserved_across_chunks` — no words are silently dropped
- `test_rejects_overlap_greater_than_or_equal_to_chunk_size` — invalid config raises instead of looping forever
