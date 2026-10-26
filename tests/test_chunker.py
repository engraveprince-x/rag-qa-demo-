import pytest

from rag.chunker import chunk_text


def test_short_text_is_single_chunk():
    chunks = chunk_text("Hello world.\n\nSecond paragraph.", "a.md", chunk_size=200, overlap=20)
    assert len(chunks) == 1
    assert chunks[0].id == "a.md#0"
    assert "Second paragraph" in chunks[0].text


def test_long_text_is_split_with_size_limit_and_overlap():
    paras = [f"Paragraph {i} " + "word " * 30 for i in range(20)]
    chunks = chunk_text("\n\n".join(paras), "b.txt", chunk_size=400, overlap=80)
    assert len(chunks) > 3
    # every chunk stays reasonably close to the limit (overlap may add a bit)
    assert all(len(c.text) <= 400 + 80 + 2 for c in chunks)
    # consecutive chunks share some text because of the overlap
    assert chunks[0].text[-40:].split()[-1] in chunks[1].text


def test_very_long_paragraph_is_hard_split():
    chunks = chunk_text("x" * 2000, "c.txt", chunk_size=500, overlap=0)
    assert len(chunks) == 4


def test_invalid_overlap():
    with pytest.raises(ValueError):
        chunk_text("abc", "d.txt", chunk_size=100, overlap=100)
