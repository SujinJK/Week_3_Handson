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
| `corpus_conflict_demo/*.md` | A copy of `corpus/` plus one extra, undated, superseded employee handbook with a different PTO policy — used only by `failure_demos.py` to demonstrate the conflicting/stale-document failure. Never touched by `ingest.py`. |
| `chunking.py` | Splits document text into overlapping word-count chunks. Pure Python, no dependencies — easy to unit test. |
| `ingest.py` | Loads `corpus/`, chunks each file, embeds chunks with a local model, stores them in a persistent Chroma collection (`chroma_db/`). |
| `rag.py` | The Q&A app: embeds the question, retrieves the top-k most similar chunks, sends them to Claude as context, prints the cited answer and which chunks were retrieved. |
| `failure_demos.py` | Deliberately triggers 3 real RAG failure modes live against the API, using disposable temporary Chroma collections — see "Failure demos" below. |
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
python failure_demos.py   # watch 3 real RAG failure modes happen live
pytest                     # unit tests for chunking
```

## Things to be aware of

- **Run order matters.** `rag.py` refuses to start ("No vector store found")
  until `ingest.py` has been run at least once.
- **`ingest.py` rebuilds from scratch every run** — it deletes and recreates
  the collection, so there's no incremental update. Re-run it after editing
  anything in `corpus/`.
- **`chroma_db/` is gitignored on purpose** — it's a local binary database,
  not source. Anyone who clones this repo (including you, on another
  machine) needs to run `ingest.py` once before `rag.py` will work.
- **Retrieval is free; only generation costs money.** Embeddings run
  entirely on your machine, no API calls — re-run `ingest.py` and
  `eval/retrieval_eval.py` as often as you want at zero cost. Only `rag.py`'s
  Claude call spends API credit, and it's small (short context, short
  answer).
- **`rag.py` has no conversation memory.** Unlike a chatbot, each question is
  answered independently with no history carried between turns — deliberate,
  to keep retrieval/citation behavior easy to reason about in isolation.
- **The corpus is intentionally tiny** (5 docs, 10 chunks) — enough to prove
  the pipeline works and to run a fast eval loop, but not representative of
  retrieval quality at real scale.
- **Embedding quality ceiling.** Local `all-MiniLM-L6-v2` performs well here
  (8/8 on the eval) because the corpus is small and topically distinct.
  Voyage AI (Anthropic's recommended embeddings provider) would generally do
  better on a larger, more ambiguous corpus — worth comparing if this grows.
- **Local environment note:** on this machine, dependencies are installed in
  a dedicated virtual environment at `C:\pyenvs\week3` (not the default
  Python) to work around a Windows path-length limit — see the Windows note
  above. Activate it (`C:\pyenvs\week3\Scripts\activate`) or call
  `C:/pyenvs/week3/Scripts/python.exe` directly when running any script here.
- **Never commit `.env`** — it holds your real `ANTHROPIC_API_KEY`. It's
  gitignored; only `.env.example` (a placeholder) is tracked.

## Embeddings and vector similarity

Each chunk of text is converted into a vector (384 numbers, from the
`all-MiniLM-L6-v2` model) that captures its meaning — chunks about similar
topics end up as vectors that point in similar directions. A question is
embedded the same way, and Chroma finds the stored chunks whose vectors are
**closest** to the question's vector — lower distance means more similar.
That's the entire "search" step: no keyword matching, so a question phrased
differently from the source text can still retrieve it.

**Parameters used, and where they live:**
- **Distance metric: squared L2 (Euclidean), not cosine.** `ingest.py`
  creates the Chroma collection without setting `metadata={"hnsw:space":
  ...}`, so it falls back to Chroma's default, which is `"l2"` — not
  `"cosine"` as an earlier version of this README incorrectly stated. For
  normalized embeddings (which `all-MiniLM-L6-v2` produces) the two metrics
  rank results identically, so retrieval quality is unaffected here, but the
  raw distance *numbers* printed by `rag.py` are squared-L2 values, not
  cosine similarity scores — don't read them as "0 to 1."
- **`chunk_size=120`, `overlap=30` words** in `chunking.py` — too
  large and irrelevant text dilutes the embedding, drowning out the specific
  fact being asked about; too small and a chunk loses the surrounding
  sentence that explains it. Overlap prevents an answer from being split
  exactly at a chunk boundary.
- **k=3** retrieved chunks per question — enough that the right chunk is
  very likely included even if it's not the single closest match, without
  flooding Claude's context with irrelevant chunks that could get cited by
  mistake.
- **Embedding model: `all-MiniLM-L6-v2`** (384 dimensions) — a small,
  general-purpose local model. Fine for this corpus's size and topic
  separation; a larger or more specialized corpus would likely benefit from
  a stronger model (e.g. Voyage AI's models).
- **`temperature` / `top_p` — not set, and not applicable here.** These are
  text-*generation* sampling knobs (how much randomness the model uses when
  writing its answer), unrelated to retrieval's `top_k`. They're easy to
  confuse because both have "top" in the name. We don't set them because
  `claude-opus-4-8` (used in `rag.py`) doesn't accept them at all — Anthropic
  removed sampling parameters starting with the 4.6 model generation in
  favor of an `effort` level, which we also leave at its default.
- **Metadata filtering: `where={"status": "current"}`.** Every chunk is
  tagged `"status": "current"` at ingest time (`ingest.py`), and `retrieve()`
  in `rag.py` filters on it *before* similarity ranking happens — a
  non-current chunk is never a candidate, not just ranked lower. Every real
  document is tagged `"current"`, so this is a no-op today (retrieval results
  are identical with or without it). It exists so that if a document is ever
  superseded, tagging it `"superseded"` excludes it from search immediately,
  without deleting the file or touching the ingest pipeline. See "Failure
  demos" below for this actually fixing the stale-document problem.

**What a production RAG system would add that this one doesn't:**
- **Reranking** — a second-stage model that re-scores the top ~20 vector
  search results for relevance before picking the final k, catching cases
  where the fast vector search's top-3 isn't actually the best 3.
- **Hybrid search** — combining vector similarity with old-fashioned keyword
  search (e.g. BM25), since embeddings can miss exact matches like product
  codes, ticket IDs, or specific terminology.
- **Query transformation** — rewriting or expanding the user's question
  before embedding it, to bridge cases where the question's wording is very
  different from the document's wording.

We skipped all three deliberately — the corpus is small and clean enough
that plain vector search already hits 8/8 on the eval set, so adding them
here would be complexity without a measurable benefit. They become worth it
as the corpus grows or gets noisier.

## Evaluation metrics

**Metric used: Hit@k (retrieval recall at k).** For each question in
`eval/eval_set.json`, checks whether the expected source document appears
anywhere among the top-k retrieved chunks. Hit@k = (questions where the
right document was found) / (total questions).

```
python -m eval.retrieval_eval
```

- **k = 3**, matching `rag.py`'s retrieval setting.
- **Current score: 8/8 (100%)** on the 8-question eval set.
- **No API calls, so it's free** — this only exercises retrieval, not
  generation.

**Why hit@k, and its limitation:** it's the simplest metric that answers
"is retrieval broken?" in isolation from generation quality — necessary
because `eval/retrieval_eval.py` deliberately never calls Claude (see "How
RAG fails" below, failure mode #1). Its main weakness is that it's
**rank-blind**: a correct document retrieved as chunk #1 scores identically
to one retrieved as chunk #3. A rank-aware metric like Mean Reciprocal Rank
(MRR) or NDCG would additionally reward retrieving the right chunk *higher*,
which hit@k can't distinguish. Not needed at this corpus size, but a real
gap if the corpus grew.

**What we did NOT measure — generation-quality metrics:**

| Metric | What it would check | Status |
|---|---|---|
| Faithfulness / groundedness | Does the answer only use retrieved content, with nothing added from outside knowledge? | Not scored — `failure_demos.py` demo 1 tests this qualitatively (does the model refuse to extrapolate), but produces no number |
| Citation accuracy | Does citation `[1]` actually support the claim next to it? | Not scored — README's "How RAG fails" #3 notes this is checked manually, and describes (but doesn't implement) an automatable check via a second Claude call |
| Answer correctness | Is the final answer actually right? | Not scored — checked manually during live runs, no automated ground-truth comparison |

Tools like RAGAS (an open-source RAG evaluation library) automate all three
via an LLM-as-judge approach. We didn't reach for one here — hit@k was
enough to validate the retrieval layer, and the failure demos cover
generation-layer risk qualitatively instead of quantitatively. Adding
automated generation scoring is the natural next step if this project grew
past a course exercise.

## Challenges RAG systems face at scale (beyond what this project shows)

This project is small and clean on purpose, so it doesn't hit every
challenge a production RAG system runs into. Worth knowing about even
though nothing here demonstrates them directly:

- **Keeping the index fresh.** If source documents change, the vector store
  goes stale until someone re-runs ingestion — at scale, "when and how do we
  re-embed changed documents" becomes its own pipeline, not a one-off script
  like `ingest.py`.
- **Evaluating quality is hard.** "Did retrieval find the right thing?" and
  "did generation answer well?" are different questions that need separate
  measurement (this is why `eval/retrieval_eval.py` deliberately tests
  retrieval alone) — and at real scale, building and maintaining a
  representative eval set is itself an ongoing project, not an 8-question
  file.
- **Cost and latency compound.** Every question costs an embedding call, a
  vector search, and a generation call. Add reranking or query
  transformation and that's more calls per question. At production volume,
  vector database hosting and embedding costs add up alongside the
  generation cost.
- **Access control.** This project pools every document into one searchable
  index. A real company has documents only some people should be able to
  retrieve (HR files vs. public docs) — plain RAG doesn't handle that by
  default; permission filtering has to be built in deliberately, typically
  by tagging chunks with access metadata and filtering at query time.

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

## Failure demos — `failure_demos.py`

Reading about failure modes is one thing; watching the pipeline actually
produce them is more convincing. This script deliberately breaks the
pipeline three different ways, live against the real API, using temporary
Chroma collections that never touch `chroma_db/` (the real pipeline's
store).

```
python failure_demos.py
```

**What actually happened when we ran it** (real transcripts, not
hypothetical):

**1. Retrieval has no "nothing is relevant" signal.** Asked
`"What is Nimbus's uptime SLA guarantee percentage?"` — a topic the corpus
never covers at all. Retrieval didn't refuse or come back empty; it
confidently returned its 3 *closest* chunks anyway (from `refund_policy.md`
and `incident_runbook.md`, neither relevant). Nearest-neighbor search always
returns *something* — "closest available" is not the same as "actually
relevant," and there's no default similarity threshold to catch the
difference. Whether a bad answer gets through from there is entirely down to
generation refusing to use context that doesn't fit — which is exactly what
our grounded system prompt is for. (In this run, Claude Opus refused
correctly with both a weak and a strict prompt — a genuinely good result,
not a failed demo. It just means this particular question wasn't enough to
trip up this particular model; the retrieval blind spot underneath it is
still real, and weaker models or trickier questions can and do fall through
it.)

**2. Chunking can fragment a fact away from itself.** The sentence "Nimbus
issues a **full refund** ... within **5 business days**" got fed through a
deliberately broken chunker (8-word chunks, no overlap). The retrieved
fragment was just `"charge within 5 business days of the report"` — missing
"full refund" entirely. Claude still recovered the number correctly, purely
by luck of where the boundary happened to fall; a slightly different chunk
size could easily have cut the fragment before "5 business days" instead,
losing the answer entirely. Real degradation, not always a *visible* one.

**3. Conflicting/stale documents produce the most dangerous failure of the
three.** We added a second, undated employee handbook with a different
(superseded) PTO policy — nothing marks it as outdated, exactly like a real
stale file nobody deleted. At **k=1** (retrieve only the single closest
chunk), the stale document happened to rank *closer* than the current one,
so the system returned: *"Unused PTO carries over up to a maximum of 10
days... [1]"* — cleanly cited, confidently worded, and **wrong**. Nothing
about that answer looks uncertain; that's what makes it the worst kind of
failure. At **k=3**, both versions get retrieved, and Claude noticed the
contradiction on its own and flagged it ("these two sources disagree,
confirm with HR") instead of picking one — a much better outcome, but one
that depends entirely on generation catching a problem retrieval should
never have surfaced unresolved.

**The fix: metadata filtering.** We tagged the stale handbook `"status":
"superseded"` and re-ran k=1 with a `where={"status": "current"}` filter
applied at query time. Result: retrieved `employee_handbook.md` (the current
one) and answered *"Unused PTO carries over up to a maximum of 5 days...
[1]"* — correct, because the stale chunk was never a candidate in the first
place, not just outranked. This is the same mechanism from "Metadata
filtering" above, now shown actually solving the problem it was added for.
The catch: the fix is only as good as the tag behind it — something still
has to mark a document `"superseded"` to begin with, which is a document
lifecycle question, not something retrieval code can solve on its own.

**The pattern across all three:** generation quality (a good system prompt,
a capable model) can catch and soften some retrieval-layer problems, but it
is a safety net, not a fix. The underlying issues — no relevance threshold,
chunk boundaries splitting facts, no document lifecycle management — live in
the retrieval layer and have to be solved there. Metadata filtering is the
one of these three we actually fixed at the retrieval layer instead of just
describing the fix.

## Terminology: what we used vs. what we skipped

**Used in this project:**

| Term | What it means | Where |
|---|---|---|
| Chunking | Splitting a document into smaller, searchable pieces | `chunking.py` |
| Embedding | Converting text into a vector of numbers that captures its meaning | `ingest.py`, `rag.py` (local, `all-MiniLM-L6-v2`) |
| Vector store / vector database | A database built to search by vector similarity, not exact match | Chroma (`chroma_db/`) |
| Persistent (vs. in-memory) store | The vector store is saved to disk and survives between runs | `chromadb.PersistentClient` |
| Similarity search / nearest-neighbor search | Finding the stored vectors closest to a query vector | `collection.query(...)` |
| Distance metric | The math used to measure "closeness" between two vectors | squared L2 (Chroma's default here) |
| `top_k` (we call it `k`) | How many of the closest results to retrieve | `k=3` in `rag.py` |
| Grounding | Restricting the model to answer only from retrieved content | `SYSTEM_PROMPT` in `rag.py` |
| Citations | Tagging each claim with which source it came from | `[1]`-style citations in answers |
| System prompt | The instruction that sets the model's behavior for the whole request | `SYSTEM_PROMPT` |
| Hallucination (as a failure mode we test for) | The model answering from outside knowledge instead of the given context | tested by the "CEO's favorite language" case and `failure_demos.py` demo 1 |
| Retrieval evaluation / hit@k | Measuring whether the right document was retrieved, separate from answer quality | `eval/retrieval_eval.py` |
| Metadata filtering | Restricting *which* chunks a query searches over, using stored metadata, before similarity ranking happens | `where={"status": "current"}` in `retrieve()` (`rag.py`); every real chunk is tagged `"current"` so it's a no-op today — `failure_demos.py` demo 3 shows it actually excluding a stale document when one exists |

**Standard RAG techniques we deliberately skipped** (see "What a production
RAG system would add" and "Challenges RAG systems face at scale" above for
why each matters and when it'd be worth adding):

| Term | What it means | Why we skipped it here |
|---|---|---|
| Reranking | A second-stage model that re-scores initial results for relevance | Corpus is small and clean enough that plain vector search already hits 8/8 on the eval |
| Hybrid search | Combining vector similarity with keyword search (e.g. BM25) | No exact-match terms (IDs, codes) in this corpus that vector search would miss |
| Query transformation / expansion / HyDE | Rewriting the question before embedding it to better match document phrasing | Questions in the eval set were already phrased close enough to the source text |
| `temperature` / `top_p` | Generation-time randomness controls (unrelated to retrieval's `top_k`, despite the similar name) | Not accepted at all by `claude-opus-4-8` — Anthropic replaced them with `effort` on recent models |
| Semantic chunking | Splitting on sentence/paragraph boundaries instead of a fixed word count | Fixed-size chunking with overlap was simpler and sufficient for this corpus; demo 2 in `failure_demos.py` shows exactly the failure semantic chunking would help avoid |
| Access control / permission-filtered retrieval | Restricting which documents a given user's queries can retrieve, e.g. `where={"allowed_roles": {"$in": [user_role]}}` | Single-user demo corpus with no real permission boundaries to enforce — the same metadata-filtering mechanism now used for `status` (see "Used in this project" above) is exactly how this would be built; we just never added an `allowed_roles` tag because there's no second user to restrict |
| Incremental indexing | Updating the vector store for changed documents only, instead of a full rebuild | `ingest.py` rebuilds from scratch every run — fine at 5 documents, not at scale |
| Agentic / multi-hop RAG | The model deciding whether/what to retrieve, and issuing further retrievals based on what it finds | Every question here is answered in a single retrieve-then-generate pass |

## Test cases

**`tests/test_chunking.py`**
- `test_empty_string_returns_no_chunks` — empty input produces no chunks
- `test_short_text_returns_single_chunk` — text shorter than chunk_size isn't split
- `test_splits_into_multiple_chunks_when_over_size` — long text is split
- `test_consecutive_chunks_overlap` — the overlap words actually repeat between chunks
- `test_all_words_are_preserved_across_chunks` — no words are silently dropped
- `test_rejects_overlap_greater_than_or_equal_to_chunk_size` — invalid config raises instead of looping forever
