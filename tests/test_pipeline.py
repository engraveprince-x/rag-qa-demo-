import io
import json
from pathlib import Path

import httpx
from pypdf import PdfWriter

from rag import RAGPipeline
from rag.llm import LLMClient, LLMConfig
from rag.loaders import extract_text

DOCS = Path(__file__).resolve().parents[1] / "data" / "docs"
NO_LLM = LLMConfig(api_key=None)


def test_sample_docs_retrieval_only_fallback():
    rag = RAGPipeline(llm_config=NO_LLM)
    rag.add_folder(DOCS)
    assert set(rag.documents) == {"store-help-center.md", "remote-work-guide.txt"}
    a = rag.ask("How many days do I have to return an item?")
    assert a.mode == "retrieval-only"
    assert a.note and "No LLM API key" in a.note
    assert a.sources[0].source == "store-help-center.md"
    assert "30 days" in a.sources[0].text
    assert a.answer.startswith("[1] store-help-center.md")
    assert "30 days" in a.answer


def test_no_results():
    rag = RAGPipeline(llm_config=NO_LLM)
    rag.add_text("x.md", "Completely unrelated content about gardening.")
    assert rag.ask("zebra quantum").mode == "no-results"


def test_reingest_replaces_document():
    rag = RAGPipeline(llm_config=NO_LLM)
    rag.add_text("doc.md", "The office plant is a cactus.")
    rag.add_text("doc.md", "The office plant is a fern.")
    a = rag.ask("office plant")
    assert len(a.sources) == 1 and "fern" in a.sources[0].text


def test_llm_mode_uses_openai_compatible_endpoint():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers["authorization"]
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "Within 30 days [1]."}}]})

    cfg = LLMConfig(api_key="test-key", base_url="http://llm.local/v1", model="demo-model")
    client = LLMClient(cfg, transport=httpx.MockTransport(handler))
    rag = RAGPipeline(llm_config=cfg, llm_client=client)
    rag.add_folder(DOCS)
    a = rag.ask("How long is the return window?")
    assert a.mode == "llm" and a.answer == "Within 30 days [1]."
    assert captured["url"] == "http://llm.local/v1/chat/completions"
    assert captured["auth"] == "Bearer test-key"
    assert captured["body"]["model"] == "demo-model"
    assert "[1] (source:" in captured["body"]["messages"][1]["content"]


def test_llm_failure_falls_back_to_retrieval():
    cfg = LLMConfig(api_key="bad", base_url="http://llm.local/v1")
    client = LLMClient(cfg, transport=httpx.MockTransport(lambda r: httpx.Response(401)))
    rag = RAGPipeline(llm_config=cfg, llm_client=client)
    rag.add_folder(DOCS)
    a = rag.ask("express shipping cost")
    assert a.mode == "retrieval-only" and "LLM call failed" in a.note


def test_pdf_loader_reads_blank_pdf():
    buf = io.BytesIO()
    w = PdfWriter()
    w.add_blank_page(width=200, height=200)
    w.write(buf)
    assert extract_text("blank.pdf", buf.getvalue()).strip() == ""


def test_fallback_extracts_matching_sentence():
    rag = RAGPipeline(llm_config=NO_LLM)
    rag.add_folder(DOCS)
    a = rag.ask("How much does express shipping cost?", top_k=1)
    assert "$14.99" in a.answer
