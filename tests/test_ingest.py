import sys
import os
import fitz  # PyMuPDF
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db.database import init_db
from core.ingest import (
    extract_text_from_pdf,
    chunk_document_pages,
    ingest_document,
    query_vector_store,
    get_indexed_stats
)


def create_sample_physics_pdf(file_path: Path) -> None:
    """Generates a multi-page sample academic PDF for testing."""
    doc = fitz.open()

    # Page 1: Classical Mechanics
    page1 = doc.new_page()
    p1_text = """CHAPTER 1: CLASSICAL MECHANICS & LAWS OF MOTION

1. Newton's First Law of Motion (Law of Inertia):
An object remains at rest or continues in uniform motion in a straight line unless acted upon by an external net force. Inertia is the inherent resistance of any physical object to any change in its velocity.

2. Newton's Second Law of Motion:
The acceleration of an object is directly proportional to the net force acting upon it and inversely proportional to its mass. Mathematically expressed as F = m * a, where F is net force in Newtons, m is mass in kg, and a is acceleration in m/s^2.

3. Newton's Third Law of Motion:
For every action, there is an equal and opposite reaction. When object A exerts a force on object B, object B simultaneously exerts an equal and opposite force on object A."""
    page1.insert_text((50, 72), p1_text, fontsize=11)

    # Page 2: Thermodynamics
    page2 = doc.new_page()
    p2_text = """CHAPTER 2: PRINCIPLES OF THERMODYNAMICS

1. The Zeroth Law of Thermodynamics:
If two thermodynamic systems are each in thermal equilibrium with a third system, they are also in thermal equilibrium with each other. This establishes temperature measurement.

2. The First Law of Thermodynamics:
Energy cannot be created or destroyed, only transformed from one form to another. The change in internal energy (Delta U) equals heat added (Q) minus work done (W): Delta U = Q - W.

3. The Second Law of Thermodynamics and Entropy:
Total entropy of an isolated system always increases over time. Heat cannot spontaneously flow from a colder body to a hotter body. The Carnot engine defines the theoretical maximum efficiency for heat engines operating between two temperatures."""
    page2.insert_text((50, 72), p2_text, fontsize=11)

    doc.save(str(file_path))
    doc.close()


def run_phase_2_tests():
    print("=" * 65)
    print("  ILA (Intelligent Learning Assistant) - Phase 2 Ingestion Test")
    print("=" * 65)

    # Initialize DB
    init_db()

    # 1. Create sample PDF
    test_pdf_path = Path("test_physics_notes.pdf")
    print(f"\n[1/5] Generating sample academic PDF: '{test_pdf_path}'...")
    create_sample_physics_pdf(test_pdf_path)
    assert test_pdf_path.exists(), "Sample PDF was not created"
    print("  -> Sample PDF generated (2 pages).")

    # 2. Test Text Extraction
    print("\n[2/5] Testing PyMuPDF Page-by-Page Extraction...")
    pages = extract_text_from_pdf(test_pdf_path, filename="test_physics_notes.pdf")
    assert len(pages) == 2, f"Expected 2 pages, got {len(pages)}"
    assert "Newton" in pages[0]["text"], "Page 1 missing Newton content"
    assert "Thermodynamics" in pages[1]["text"], "Page 2 missing Thermodynamics content"
    print(f"  -> Extracted {len(pages)} pages successfully with clean formatting.")

    # 3. Test Full Ingestion Pipeline
    print("\n[3/5] Ingesting Document into Persistent ChromaDB...")
    def progress_printer(pct, total, msg):
        print(f"     [{pct:3d}%] {msg}")

    result = ingest_document(test_pdf_path, filename="test_physics_notes.pdf", progress_callback=progress_printer)
    assert result["status"] in ("success", "cached"), f"Ingestion failed: {result}"
    print(f"  -> Ingestion completed: {result['message']}")

    # 4. Test Deduplication Caching
    print("\n[4/5] Testing SHA-256 Deduplication Cache...")
    cached_result = ingest_document(test_pdf_path, filename="test_physics_notes.pdf")
    assert cached_result["status"] == "cached", f"Expected cached status, got {cached_result['status']}"
    print(f"  -> Deduplication cache verified: {cached_result['message']}")

    # 5. Test Semantic Retrieval Queries
    print("\n[5/5] Testing Semantic Vector Retrieval from ChromaDB...")
    
    # Query 1: Law of Inertia
    q1 = "What is the law of inertia and how does an object behave?"
    print(f"  Query A: \"{q1}\"")
    results1 = query_vector_store(q1, top_k=2)
    assert len(results1) > 0, "No results returned for Query A"
    top_match_1 = results1[0]
    print(f"     Top Match Source: {top_match_1['metadata']['source_file']}, Page {top_match_1['metadata']['page_number']}")
    print(f"     Similarity Score: {top_match_1['similarity_score']} (Distance: {top_match_1['distance']:.4f})")
    print(f"     Snippet: \"{top_match_1['text'][:100]}...\"")
    assert top_match_1['metadata']['page_number'] == 1, "Expected page 1 for Newton's laws"

    # Query 2: Entropy and Carnot
    q2 = "Explain entropy and maximum efficiency in heat engines"
    print(f"\n  Query B: \"{q2}\"")
    results2 = query_vector_store(q2, top_k=2)
    assert len(results2) > 0, "No results returned for Query B"
    top_match_2 = results2[0]
    print(f"     Top Match Source: {top_match_2['metadata']['source_file']}, Page {top_match_2['metadata']['page_number']}")
    print(f"     Similarity Score: {top_match_2['similarity_score']} (Distance: {top_match_2['distance']:.4f})")
    print(f"     Snippet: \"{top_match_2['text'][:100]}...\"")
    assert top_match_2['metadata']['page_number'] == 2, "Expected page 2 for Thermodynamics"

    # Verify overall collection stats
    stats = get_indexed_stats()
    print(f"\nIndexed Knowledge Base Stats: {stats['total_documents']} documents, {stats['total_chunks_in_chroma']} total chunks in ChromaDB.")

    print("\n" + "=" * 65)
    print("  ALL PHASE 2 INGESTION & RETRIEVAL TESTS PASSED!")
    print("=" * 65)


if __name__ == "__main__":
    try:
        run_phase_2_tests()
    except Exception as e:
        print(f"\n[ERROR] Phase 2 test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
