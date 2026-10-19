"""
ILA - Ask Any Document (CLI Interface)
Allows indexing any custom PDF and asking questions from it immediately.
"""

import sys
import os
from pathlib import Path

# UTF-8 stdout configuration for Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.ingest import ingest_document, query_vector_store, get_indexed_stats
from core.rag import query_rag_stream, format_retrieved_context


def main():
    print("=" * 65)
    print("  INTELLIGENT LEARNING ASSISTANT (ILA) - STUDY MATERIAL Q&A")
    print("  100% Offline | Model: Qwen3:8b | Persistent ChromaDB")
    print("=" * 65)

    pdf_arg = sys.argv[1] if len(sys.argv) > 1 else None

    if not pdf_arg:
        stats = get_indexed_stats()
        print(f"\nCurrently Indexed: {stats['total_documents']} document(s), {stats['total_chunks_in_chroma']} chunks.")
        user_input = input("\nEnter PDF path to index (or press ENTER to use existing notes): ").strip()
        # Strip quotation marks if user dragged-and-dropped the file into terminal
        user_input = user_input.strip('"').strip("'")
        if user_input:
            pdf_arg = user_input

    # If a PDF path was provided, ingest it now
    if pdf_arg:
        pdf_path = Path(pdf_arg)
        if not pdf_path.exists():
            print(f"[Error] File not found: {pdf_path}")
            return

        print(f"\n[Ingesting] '{pdf_path.name}'...")
        def progress(pct, total, msg):
            print(f"  [{pct:3d}%] {msg}")

        try:
            res = ingest_document(pdf_path, filename=pdf_path.name, progress_callback=progress)
            print(f"\n-> {res['message']}")
        except Exception as e:
            print(f"[Error during indexing]: {e}")
            return

    # Enter Question Loop
    print("\n" + "=" * 65)
    print("Ready for questions! (Type 'exit' to quit)")
    print("=" * 65)

    while True:
        try:
            q = input("\nQuestion > ").strip()
            if not q:
                continue
            if q.lower() in ("exit", "quit", "q"):
                print("Exiting. Good luck with your meeting!")
                break

            print("\nILA Answer: ", end="", flush=True)
            full_response = ""
            for token in query_rag_stream(q):
                full_response += token
                print(token, end="", flush=True)
            print("\n")

            # Print source citation only if the answer was actually grounded
            chunks = query_vector_store(q, top_k=2)
            is_unanswered = "don't have enough information" in full_response or "not in your uploaded" in full_response
            
            if chunks and not is_unanswered and chunks[0].get("similarity_score", 0) >= 0.50:
                _, sources = format_retrieved_context(chunks)
                if sources:
                    print("-" * 50)
                    print(f"📌 Citation: {sources[0]['source_file']} (Page {sources[0]['page_number']}) | Relevance: {sources[0]['similarity_score'] * 100:.1f}%")
                    print("-" * 50)

        except KeyboardInterrupt:
            print("\nExiting.")
            break
        except Exception as e:
            print(f"\n[Error]: {e}")


if __name__ == "__main__":
    main()
