from rag.chunker import Chunk
from rag.retriever import BM25Retriever, tokenize


def _chunks():
    texts = {
        "shipping": "Standard shipping takes 3 to 5 business days. Express shipping is faster.",
        "returns": "You can return items within 30 days for a refund to the original payment method.",
        "cn": "退货政策：商品在签收后三十天内可以申请退货退款。",
    }
    return [Chunk(id=f"{k}#0", source=k, text=v, index=0) for k, v in texts.items()]


def test_tokenize_handles_english_and_chinese():
    toks = tokenize("How do I RETURN items? 退货政策")
    assert "return" in toks and "items" in toks
    assert "how" not in toks  # stopword
    assert "退货" in toks and "政策" in toks


def test_bm25_ranks_relevant_chunk_first():
    r = BM25Retriever()
    r.add(_chunks())
    assert r.search("how long does shipping take")[0].chunk.source == "shipping"
    assert r.search("refund policy for returns")[0].chunk.source == "returns"
    assert r.search("怎么退货")[0].chunk.source == "cn"


def test_bm25_no_match_and_remove():
    r = BM25Retriever()
    r.add(_chunks())
    assert r.search("quantum chromodynamics") == []
    assert r.remove_source("shipping") == 1
    assert all(res.chunk.source != "shipping" for res in r.search("shipping"))
