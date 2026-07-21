"""Two chunking strategies for splitting document text into searchable pieces.

chunk_text() is a fixed word-count chunker: simple, but it can and does cut a
sentence in half at a chunk boundary (see failure_demos.py demo 2).

semantic_chunk_text() splits on sentence boundaries first, then groups whole
sentences into chunks up to a target size -- a chunk boundary can only ever
fall between sentences, never inside one. This is what ingest.py actually
uses; chunk_text() is kept for tests and for failure_demos.py's comparison.
"""
import re

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def chunk_text(text: str, chunk_size: int = 120, overlap: int = 30) -> list[str]:
    """Split text into chunks of chunk_size words, each overlapping the previous by overlap words."""
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = end - overlap
    return chunks


def _split_sentences(text: str) -> list[str]:
    """Split whitespace-normalized text into sentences on '.', '!', or '?' + whitespace."""
    normalized = " ".join(text.split())
    if not normalized:
        return []
    return [s for s in _SENTENCE_BOUNDARY.split(normalized) if s]


def semantic_chunk_text(text: str, max_chunk_size: int = 120) -> list[str]:
    """Group whole sentences into chunks of up to max_chunk_size words each.

    A chunk boundary only ever falls between sentences -- never inside one --
    so a fact and the sentence that supports it can never be split apart the
    way chunk_text() can split them. The tradeoff: a single sentence longer
    than max_chunk_size becomes its own over-sized chunk rather than being
    cut, since a sentence is the smallest unit this function will break.
    """
    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    for sentence in sentences:
        sentence_words = len(sentence.split())
        if current and current_words + sentence_words > max_chunk_size:
            chunks.append(" ".join(current))
            current = []
            current_words = 0
        current.append(sentence)
        current_words += sentence_words

    if current:
        chunks.append(" ".join(current))

    return chunks
