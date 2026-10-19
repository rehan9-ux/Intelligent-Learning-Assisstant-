import sqlite3
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from core.config import SQLITE_DB_PATH

SCHEMA_VERSION = 1


def get_db_connection() -> sqlite3.Connection:
    """Creates and returns a SQLite database connection with Row factory."""
    conn = sqlite3.connect(str(SQLITE_DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Initializes the SQLite database schema and indexes."""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Schema version tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL
            )
        """)

        # Indexed Documents Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS indexed_documents (
                file_hash TEXT PRIMARY KEY,
                file_name TEXT NOT NULL,
                chunk_count INTEGER NOT NULL,
                indexed_at TEXT NOT NULL
            )
        """)

        # Quiz Results Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS quiz_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL,
                topic TEXT NOT NULL,
                question TEXT NOT NULL,
                options_json TEXT,
                selected_option TEXT,
                correct_answer TEXT,
                is_correct INTEGER NOT NULL,
                explanation TEXT,
                timestamp TEXT NOT NULL
            )
        """)

        # Index on student_id and topic for fast mastery lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_student_topic 
            ON quiz_results(student_id, topic)
        """)

        # Student Profiles Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS student_profiles (
                student_id TEXT PRIMARY KEY,
                level TEXT DEFAULT 'beginner',
                explanation_style TEXT DEFAULT 'analogies',
                mastery_json TEXT DEFAULT '{}',
                updated_at TEXT NOT NULL
            )
        """)

        # Spaced Repetition (SM-2) Tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS spaced_repetition (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL,
                topic TEXT NOT NULL,
                ease_factor REAL DEFAULT 2.5,
                interval INTEGER DEFAULT 1,
                repetitions INTEGER DEFAULT 0,
                next_review_date TEXT,
                last_reviewed_at TEXT,
                UNIQUE(student_id, topic)
            )
        """)

        # Chat History Table (Logs all student queries, AI answers, citations, and timestamps)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                sources_json TEXT DEFAULT '[]',
                timestamp TEXT NOT NULL
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_chat_student 
            ON chat_history(student_id, timestamp)
        """)

        # Record schema version if not present
        cursor.execute("SELECT version FROM schema_version WHERE version = ?", (SCHEMA_VERSION,))
        if not cursor.fetchone():
            cursor.execute(
                "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
                (SCHEMA_VERSION, datetime.now().isoformat())
            )

        conn.commit()


# --- Document Tracking Helpers ---

def is_document_indexed(file_hash: str) -> bool:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM indexed_documents WHERE file_hash = ?", (file_hash,))
        return cursor.fetchone() is not None


def record_indexed_document(file_hash: str, file_name: str, chunk_count: int) -> None:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO indexed_documents (file_hash, file_name, chunk_count, indexed_at)
            VALUES (?, ?, ?, ?)
        """, (file_hash, file_name, chunk_count, datetime.now().isoformat()))
        conn.commit()


def get_indexed_documents() -> List[Dict[str, Any]]:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT file_hash, file_name, chunk_count, indexed_at FROM indexed_documents ORDER BY indexed_at DESC")
        return [dict(row) for row in cursor.fetchall()]


# --- Student Profile Helpers ---

def get_or_create_profile(student_id: str) -> Dict[str, Any]:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM student_profiles WHERE student_id = ?", (student_id,))
        row = cursor.fetchone()
        if row:
            res = dict(row)
            res["mastery"] = json.loads(res.get("mastery_json") or "{}")
            return res

        # Create default
        default_profile = {
            "student_id": student_id,
            "level": "beginner",
            "explanation_style": "analogies",
            "mastery_json": "{}",
            "updated_at": datetime.now().isoformat()
        }
        cursor.execute("""
            INSERT INTO student_profiles (student_id, level, explanation_style, mastery_json, updated_at)
            VALUES (:student_id, :level, :explanation_style, :mastery_json, :updated_at)
        """, default_profile)
        conn.commit()
        default_profile["mastery"] = {}
        return default_profile


def update_student_profile(student_id: str, level: str, explanation_style: str, mastery_dict: Dict[str, float]) -> None:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE student_profiles 
            SET level = ?, explanation_style = ?, mastery_json = ?, updated_at = ?
            WHERE student_id = ?
        """, (level, explanation_style, json.dumps(mastery_dict), datetime.now().isoformat(), student_id))
        conn.commit()


def reset_student_profile(student_id: str) -> None:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM quiz_results WHERE student_id = ?", (student_id,))
        cursor.execute("DELETE FROM student_profiles WHERE student_id = ?", (student_id,))
        cursor.execute("DELETE FROM spaced_repetition WHERE student_id = ?", (student_id,))
        conn.commit()


# --- Quiz Results Helpers ---

def record_quiz_result(
    student_id: str,
    topic: str,
    question: str,
    options: List[str],
    selected_option: str,
    correct_answer: str,
    is_correct: bool,
    explanation: str = ""
) -> None:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO quiz_results (
                student_id, topic, question, options_json, selected_option, 
                correct_answer, is_correct, explanation, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            student_id, topic, question, json.dumps(options), selected_option,
            correct_answer, 1 if is_correct else 0, explanation, datetime.now().isoformat()
        ))
        conn.commit()


def get_quiz_history(student_id: str, topic: Optional[str] = None) -> List[Dict[str, Any]]:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        if topic:
            cursor.execute("""
                SELECT * FROM quiz_results 
                WHERE student_id = ? AND topic = ? 
                ORDER BY id ASC
            """, (student_id, topic))
        else:
            cursor.execute("""
                SELECT * FROM quiz_results 
                WHERE student_id = ? 
                ORDER BY id ASC
            """, (student_id,))
        return [dict(row) for row in cursor.fetchall()]


# --- Chat History Helpers ---

def record_chat_message(
    student_id: str,
    question: str,
    answer: str,
    sources: Optional[List[Dict[str, Any]]] = None
) -> None:
    """Logs student question, AI answer, and citations to SQLite."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO chat_history (student_id, question, answer, sources_json, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, (
            student_id,
            question,
            answer,
            json.dumps(sources or []),
            datetime.now().isoformat()
        ))
        conn.commit()


def get_chat_history(student_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieves chronological chat history for a student."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, student_id, question, answer, sources_json, timestamp 
            FROM chat_history 
            WHERE student_id = ? 
            ORDER BY id ASC 
            LIMIT ?
        """, (student_id, limit))
        rows = cursor.fetchall()
        results = []
        for row in rows:
            r = dict(row)
            r["sources"] = json.loads(r.get("sources_json") or "[]")
            results.append(r)
        return results
