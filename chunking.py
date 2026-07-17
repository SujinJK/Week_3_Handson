"""Splits document text into overlapping word-count chunks.

Word-based (not character-based) so chunk boundaries fall on whole words.
Overlap keeps a sliding window of context so an answer near a chunk boundary
isn't split away from the sentence that explains it.
"""


def chunk_text(text: str, chunk_size: int = 120, overlap: int = 30) -> list[str]:
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
