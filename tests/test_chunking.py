import pytest

from chunking import chunk_text


class TestChunkText:
    def test_empty_string_returns_no_chunks(self):
        assert chunk_text("") == []

    def test_short_text_returns_single_chunk(self):
        text = "one two three"
        assert chunk_text(text, chunk_size=10, overlap=2) == ["one two three"]

    def test_splits_into_multiple_chunks_when_over_size(self):
        words = [f"word{i}" for i in range(10)]
        text = " ".join(words)
        chunks = chunk_text(text, chunk_size=4, overlap=1)
        assert len(chunks) > 1

    def test_consecutive_chunks_overlap(self):
        words = [f"word{i}" for i in range(10)]
        text = " ".join(words)
        chunks = chunk_text(text, chunk_size=4, overlap=2)
        first_words = chunks[0].split()
        second_words = chunks[1].split()
        # last `overlap` words of chunk 1 should equal the first `overlap` words of chunk 2
        assert first_words[-2:] == second_words[:2]

    def test_all_words_are_preserved_across_chunks(self):
        words = [f"word{i}" for i in range(23)]
        text = " ".join(words)
        chunks = chunk_text(text, chunk_size=5, overlap=1)
        seen = set()
        for chunk in chunks:
            seen.update(chunk.split())
        assert seen == set(words)

    def test_rejects_overlap_greater_than_or_equal_to_chunk_size(self):
        with pytest.raises(ValueError):
            chunk_text("a b c d e f", chunk_size=3, overlap=3)
