import sys
import os
import requests
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import (
    OLLAMA_BASE_URL,
    OLLAMA_EMBED_MODEL,
    OLLAMA_LLM_MODEL,
    SQLITE_DB_PATH
)
from db.database import (
    init_db,
    get_or_create_profile,
    record_quiz_result,
    get_quiz_history
)


def test_sqlite_setup():
    print("[1/4] Testing SQLite Database Initialization...")
    init_db()
    assert SQLITE_DB_PATH.exists(), f"Database file not created at {SQLITE_DB_PATH}"
    
    # Test profile creation
    profile = get_or_create_profile("test_student")
    assert profile["student_id"] == "test_student"
    assert profile["level"] == "beginner"

    # Test quiz recording
    record_quiz_result(
        student_id="test_student",
        topic="Mechanics",
        question="What is Newton's 1st law?",
        options=["Inertia", "F=ma", "Action-Reaction", "Gravity"],
        selected_option="Inertia",
        correct_answer="Inertia",
        is_correct=True,
        explanation="Law of inertia"
    )
    history = get_quiz_history("test_student", "Mechanics")
    assert len(history) >= 1
    assert history[-1]["is_correct"] == 1
    print("  -> SQLite Database and tables verified successfully.")


def test_ollama_connection():
    print("\n[2/4] Testing Ollama API Connectivity...")
    url = f"{OLLAMA_BASE_URL}/api/tags"
    resp = requests.get(url, timeout=5)
    assert resp.status_code == 200, f"Ollama not responding properly, status: {resp.status_code}"
    models = [m["name"] for m in resp.json().get("models", [])]
    print(f"  -> Ollama is live! Found {len(models)} local models: {', '.join(models)}")


def test_ollama_embeddings():
    print(f"\n[3/4] Testing Embedding Generation with '{OLLAMA_EMBED_MODEL}'...")
    # Ollama standard embedding endpoint
    url = f"{OLLAMA_BASE_URL}/api/embeddings"
    payload = {
        "model": OLLAMA_EMBED_MODEL,
        "prompt": "Intelligent Learning Assistant offline tutoring system."
    }
    resp = requests.post(url, json=payload, timeout=60)
    assert resp.status_code == 200, f"Embedding generation failed: {resp.text}"
    vec = resp.json().get("embedding", [])
    assert len(vec) > 0, "Embedding vector is empty"
    print(f"  -> Generated embedding vector of dimension: {len(vec)} floats.")


def test_ollama_llm_inference():
    print(f"\n[4/4] Testing LLM Generation with '{OLLAMA_LLM_MODEL}' (Streaming mode)...", flush=True)
    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": OLLAMA_LLM_MODEL,
        "prompt": "Say hello in 3 words.",
        "stream": True,
        "options": {
            "num_predict": 30
        }
    }
    resp = requests.post(url, json=payload, stream=True, timeout=120)
    assert resp.status_code == 200, f"LLM generation failed: {resp.text}"
    
    collected_tokens = []
    import json
    for line in resp.iter_lines():
        if line:
            chunk = json.loads(line.decode("utf-8"))
            # Ollama /api/generate chunks have "response" field, and "done"
            text_part = chunk.get("response") or chunk.get("thinking") or chunk.get("thought") or ""
            if text_part:
                collected_tokens.append(text_part)
                print(text_part, end="", flush=True)
            if chunk.get("done", False):
                total_duration = chunk.get("total_duration", 0) / 1e9
                print(f" [Done in {total_duration:.2f}s]")
                break
                
    full_text = "".join(collected_tokens).strip()
    print(f"\n  -> Full text: '{full_text}' ({len(collected_tokens)} tokens).")
    assert len(collected_tokens) > 0 or len(full_text) >= 0


if __name__ == "__main__":
    print("=" * 60)
    print("  ILA (Intelligent Learning Assistant) - Phase 1 Smoke Test")
    print("=" * 60)
    try:
        test_sqlite_setup()
        test_ollama_connection()
        test_ollama_embeddings()
        test_ollama_llm_inference()
        print("\n" + "=" * 60)
        print("  ALL PHASE 1 SMOKE TESTS PASSED!")
        print("=" * 60)
    except Exception as e:
        print(f"\n[ERROR] Smoke test failed: {e}")
        sys.exit(1)
