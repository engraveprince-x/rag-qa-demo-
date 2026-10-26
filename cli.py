"""Command-line interface.

    python cli.py ask "How do returns work?" --docs data/docs
    python cli.py chat --docs path/to/folder
"""

from __future__ import annotations

import argparse
import json

from rag import RAGPipeline


def main() -> None:
    p = argparse.ArgumentParser(description="Ask questions about a folder of documents.")
    p.add_argument("command", choices=["ask", "chat"])
    p.add_argument("question", nargs="?")
    p.add_argument("--docs", default="data/docs", help="folder with .txt/.md/.pdf files")
    p.add_argument("--top-k", type=int, default=4)
    p.add_argument("--retriever", choices=["bm25", "embedding"], default="bm25")
    p.add_argument("--json", action="store_true", help="print raw JSON")
    args = p.parse_args()

    rag = RAGPipeline(retriever_kind=args.retriever)
    n = rag.add_folder(args.docs)
    print(f"Indexed {len(rag.documents)} document(s), {n} chunk(s). LLM: {'on' if rag.llm else 'off'}\n")

    def run(q: str) -> None:
        a = rag.ask(q, top_k=args.top_k)
        if args.json:
            print(json.dumps(a.to_dict(), ensure_ascii=False, indent=2))
            return
        print(a.answer)
        if a.note:
            print(f"\n({a.note})")
        if a.mode == "llm":
            print("\nSources: " + ", ".join(f"[{s.n}] {s.chunk_id}" for s in a.sources))
        print()

    if args.command == "ask":
        if not args.question:
            p.error("ask requires a question")
        run(args.question)
    else:
        while True:
            try:
                q = input("You> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if q in {"exit", "quit"}:
                break
            if q:
                run(q)


if __name__ == "__main__":
    main()
