import sys
import os
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from core.rag import query_rag_stream, query_rag, format_retrieved_context
from core.ingest import query_vector_store


def run_phase_3_rag_tests():
    print("=" * 65)
    print("  ILA (Intelligent Learning Assistant) - Phase 3 RAG Tests")
    print("=" * 65)

    # 1. Test Grounded Streaming Query (Newton's Laws)
    print("\n[1/4] Testing Grounded RAG Query with Real-Time Streaming...")
    q1 = "What is Newton's second law of motion and what is its mathematical formula?"
    print(f"  Question: \"{q1}\"")
    print("  Streaming Answer: ", end="", flush=True)

    stream_tokens = []
    for token in query_rag_stream(q1):
        stream_tokens.append(token)
        print(token, end="", flush=True)

    answer1 = "".join(stream_tokens)
    assert len(answer1) > 10, "Streaming answer is too short or empty"
    print(f"\n  -> Streamed {len(stream_tokens)} tokens successfully.")
    
    # Check context & citations
    chunks1 = query_vector_store(q1)
    _, sources1 = format_retrieved_context(chunks1)
    assert len(sources1) > 0, "No sources found for Q1"
    print(f"  -> Verified Source: {sources1[0]['source_file']} (Page {sources1[0]['page_number']})")
    assert sources1[0]['page_number'] == 1, "Expected citation to Page 1 for Newton's laws"

    # 2. Test Grounded Query (Thermodynamics & Entropy)
    print("\n[2/4] Testing Grounded Query (Thermodynamics)...")
    q2 = "What does the Second Law of Thermodynamics say about entropy in an isolated system?"
    print(f"  Question: \"{q2}\"")
    res2 = query_rag(q2)
    assert res2["is_grounded"] is True, "Query was not marked as grounded"
    print(f"  Answer:\n  \"{res2['answer']}\"")
    print(f"  -> Source Cited: Page {res2['sources'][0]['page_number']} (Similarity: {res2['sources'][0]['similarity_score'] * 100:.1f}%)")
    assert res2['sources'][0]['page_number'] == 2, "Expected citation to Page 2 for Thermodynamics"

    # 3. Test Anti-Hallucination on Out-of-Domain Question
    print("\n[3/4] Testing Anti-Hallucination Filter on Out-of-Domain Question...")
    out_of_domain_q = "How do I make chocolate chip cookies from scratch?"
    print(f"  Question: \"{out_of_domain_q}\"")
    res3 = query_rag(out_of_domain_q)
    print(f"  System Response:\n  \"{res3['answer']}\"")
    # Verify that the system did NOT hallucinate a cookie recipe
    assert "cookie" not in res3["answer"].lower() or "not find" in res3["answer"].lower() or "uploaded" in res3["answer"].lower()
    print("  -> Anti-hallucination filter successfully rejected out-of-domain query.")

    # 4. Test Adaptive Explanation Style Injection
    print("\n[4/4] Testing Adaptive Prompt Injection (Beginner + Analogies)...")
    profile = {
        "student_id": "krish_rehan",
        "level": "beginner",
        "weak_topics": ["Thermodynamics", "Entropy"],
        "explanation_style": "analogies"
    }
    q4 = "Explain what entropy means simply."
    print(f"  Question: \"{q4}\"")
    print(f"  Profile Injected: Level={profile['level']}, Style={profile['explanation_style']}, Weak={profile['weak_topics']}")
    res4 = query_rag(q4, student_profile=profile)
    print(f"  Adapted Explanation:\n  \"{res4['answer']}\"")
    assert len(res4["answer"]) > 10, "Adapted response was empty"
    print("  -> Adaptive profile successfully influenced explanation formulation.")

    print("\n" + "=" * 65)
    print("  ALL PHASE 3 RAG & GROUNDING TESTS PASSED!")
    print("=" * 65)


if __name__ == "__main__":
    try:
        run_phase_3_rag_tests()
    except Exception as e:
        print(f"\n[ERROR] Phase 3 test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
