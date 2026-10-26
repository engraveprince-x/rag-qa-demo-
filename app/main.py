"""FastAPI web app: upload documents, list them, and ask questions."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from rag import RAGPipeline
from rag.loaders import SUPPORTED_EXTENSIONS, UnsupportedFileType

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = Path(os.getenv("RAG_DOCS_DIR", BASE_DIR.parent / "data" / "docs"))
MAX_UPLOAD_BYTES = int(os.getenv("RAG_MAX_UPLOAD_MB", "10")) * 1024 * 1024


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(4, ge=1, le=10)


def create_app(pipeline: RAGPipeline | None = None, preload: bool = True) -> FastAPI:
    rag = pipeline or RAGPipeline(retriever_kind=os.getenv("RAG_RETRIEVER", "bm25"))
    if preload and DOCS_DIR.exists():
        rag.add_folder(DOCS_DIR)

    app = FastAPI(title="Knowledge Base Q&A (RAG demo)", version="1.0.0")
    app.state.rag = rag
    app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(BASE_DIR / "static" / "index.html")

    @app.get("/api/health")
    def health():
        return {
            "status": "ok",
            "retriever": rag.retriever.name,
            "llm_enabled": rag.llm is not None,
            "llm_model": rag.llm_config.model if rag.llm is not None else None,
        }

    @app.get("/api/documents")
    def list_documents():
        return {"documents": [{"source": s, "chunks": n} for s, n in sorted(rag.documents.items())]}

    @app.post("/api/documents")
    async def upload(files: list[UploadFile] = File(...)):
        added = []
        for f in files:
            name = Path(f.filename or "").name
            if Path(name).suffix.lower() not in SUPPORTED_EXTENSIONS:
                raise HTTPException(400, f"Unsupported file type: {name}")
            data = await f.read()
            if len(data) > MAX_UPLOAD_BYTES:
                raise HTTPException(413, f"File too large: {name}")
            try:
                n = rag.add_file_bytes(name, data)
            except UnsupportedFileType as exc:
                raise HTTPException(400, str(exc))
            except Exception as exc:
                raise HTTPException(422, f"Could not read {name}: {exc}")
            added.append({"source": name, "chunks": n})
        return {"added": added}

    @app.delete("/api/documents/{source}")
    def delete(source: str):
        if not rag.remove(source):
            raise HTTPException(404, "Document not found")
        return {"deleted": source}

    @app.post("/api/ask")
    def ask(req: AskRequest):
        try:
            return rag.ask(req.question, top_k=req.top_k).to_dict()
        except ValueError as exc:
            raise HTTPException(400, str(exc))

    return app


app = create_app()
