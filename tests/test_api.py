from fastapi.testclient import TestClient

from app.main import create_app
from rag import RAGPipeline
from rag.llm import LLMConfig


def _client():
    return TestClient(create_app(RAGPipeline(llm_config=LLMConfig(api_key=None)), preload=True))


def test_health_and_index_page():
    c = _client()
    h = c.get("/api/health").json()
    assert h == {"status": "ok", "retriever": "bm25", "llm_enabled": False, "llm_model": None}
    assert "Knowledge Base" in c.get("/").text


def test_upload_ask_delete_flow():
    c = _client()
    files = [("files", ("faq.txt", b"The secret project codename is Bluebird.", "text/plain"))]
    r = c.post("/api/documents", files=files)
    assert r.status_code == 200 and r.json()["added"][0]["source"] == "faq.txt"

    a = c.post("/api/ask", json={"question": "What is the project codename?"}).json()
    assert a["sources"][0]["source"] == "faq.txt"
    assert a["mode"] == "retrieval-only"

    assert c.delete("/api/documents/faq.txt").status_code == 200
    docs = [d["source"] for d in c.get("/api/documents").json()["documents"]]
    assert "faq.txt" not in docs


def test_rejects_bad_input():
    c = _client()
    r = c.post("/api/documents", files=[("files", ("evil.exe", b"MZ", "application/octet-stream"))])
    assert r.status_code == 400
    assert c.post("/api/ask", json={"question": ""}).status_code == 422
    assert c.delete("/api/documents/missing.md").status_code == 404
