# Intelligent Learning Assistant (ILA)
## Project Codebase Architecture & File-by-File Technical Guide

**Prepared for**: Faculty Advisor / Professor Review Meeting  
**Project Title**: Intelligent Learning Assistant (ILA) — Offline Adaptive AI Study Assistant  
**Team Members**: Krish Shandilya (24UCS156) & Rehan Khan (24UCS277)  
**Degree**: B.Tech Computer Science & Engineering  

---

## 1. Executive Summary & Current Status

**ILA** is a completely self-contained, offline AI tutoring system designed to run on consumer hardware with zero internet access, zero cloud API fees, and strict data privacy. It unifies three core computer science paradigms:
1. **Retrieval-Augmented Generation (RAG)** using local vector embeddings.
2. **Offline Generative AI** using an open-weights LLM (`qwen3:8b`) served via Ollama.
3. **Cognitive Machine Learning** using Bayesian Knowledge Tracing (BKT) and the SM-2 Spaced Repetition algorithm to mathematically track and adapt to individual student knowledge states.

### Status Snapshot:
* ✅ **Phase 1: Environment & Offline Runtime Setup** (Completed & Test-Verified)
* ✅ **Phase 2: Document Ingestion Pipeline** (Completed & Test-Verified)
* ✅ **Phase 3: RAG Query Engine & Grounded Generation** (Completed & Test-Verified)
* ⏳ **Phase 4: Quiz Generation & Bayesian Knowledge Tracing** (Next Step)
* ⏳ **Phase 5: Adaptive Explanation Engine & Spaced Repetition** (Upcoming)
* ⏳ **Phase 6: Streamlit Interactive UI & Local LAN Server** (Upcoming)

---

## 2. Full Project Directory Structure

```text
MINI PROJECT/
│
├── requirements.txt                         # Pinned Python package dependencies
├── test_physics_notes.pdf                   # Test academic PDF generated for verification
├── an Offline Adaptive AI Study Assist.txt  # Initial project concept brief & team details
│
├── ILA_Build_Workflow.pdf                   # Phase-by-phase build blueprint
├── ILA_Detailed_Workflow.pdf                # Detailed recommendations, risks & best practices
├── ILA_Offline_Model_Guide.pdf              # Offline LLM & embedding model benchmark guide
├── ILA_STUDY_GUIDE_AND_VIVA_PREP.md         # Comprehensive viva prep & theory reference
├── PROJECT_STRUCTURE_AND_CODEBASE_GUIDE.md  # [This Document] File-by-file codebase guide
│
├── core/                                    # Core AI & Ingestion Logic
│   ├── __init__.py                          # Package initializer
│   ├── config.py                            # Centralized system configurations & parameters
│   ├── prompts.py                           # Versioned system prompts & adaptive templates
│   ├── ingest.py                            # PDF parsing, chunking, caching & ChromaDB indexing
│   └── rag.py                               # RAG query retrieval, streaming & citations
│
├── db/                                      # Relational Database Persistence Layer
│   ├── __init__.py                          # Package initializer
│   └── database.py                          # SQLite schemas, B-tree indexing & CRUD helpers
│
├── data/                                    # Local Persistent Data Storage
│   ├── ila_students.db                      # SQLite relational database (profiles, quizzes)
│   └── ila_db/                              # ChromaDB persistent vector database (embeddings)
│
├── tests/                                   # Automated Verification Test Suite
│   ├── __init__.py                          # Package initializer
│   ├── test_smoke.py                        # Phase 1: Offline runtime & model connectivity test
│   ├── test_ingest.py                       # Phase 2: PDF ingestion, chunking & ChromaDB test
│   └── test_rag.py                          # Phase 3: Grounded RAG, streaming & anti-hallucination test
│
└── ila_env/                                 # Isolated Python 3.14 Virtual Environment
```

---

## 3. Detailed File-by-File Breakdown

### Root Configuration & Reference Files

#### 1. [`requirements.txt`](file:///c:/Users/rehan/OneDrive/Desktop/MINI%20PROJECT/requirements.txt)
* **Purpose**: Declares all third-party Python libraries required to run ILA offline.
* **Key Packages**:
  * `chromadb`: Local serverless vector database with HNSW similarity search.
  * `langchain` & `langchain-text-splitters`: Document splitting and chunk boundary management.
  * `pymupdf` (`fitz`): High-speed PDF parser preserving document structure and page coordinates.
  * `streamlit` & `plotly`: Interactive frontend UI and visual student mastery dashboards.
  * `fastapi` & `uvicorn`: High-performance asynchronous REST API backend.
  * `requests`: Local HTTP communication with the Ollama service on `localhost:11434`.
  * `pandas` & `numpy`: Tabular data processing and numerical computations for BKT.

#### 2. [`ILA_STUDY_GUIDE_AND_VIVA_PREP.md`](file:///c:/Users/rehan/OneDrive/Desktop/MINI%20PROJECT/ILA_STUDY_GUIDE_AND_VIVA_PREP.md)
* **Purpose**: Comprehensive student study guide and viva exam handbook.
* **Contents**:
  * Mathematical formulation of Cosine Similarity and Cosine Distance.
  * Bayesian Knowledge Tracing (BKT) 4-parameter Hidden Markov Model formulation ($P(L_0), P(T), P(G), P(S)$).
  * SM-2 Spaced Repetition formulas and Ebbinghaus forgetting curve modeling.
  * Top 10 B.Tech viva interview questions with model answers.

---

### The `core/` Package (Business Logic & AI Engine)

#### 3. [`core/config.py`](file:///c:/Users/rehan/OneDrive/Desktop/MINI%20PROJECT/core/config.py)
* **Role**: **Centralized Configuration and Hyperparameter Registry**.
* **What it does**:
  * Defines filesystem paths for data directories (`./data/ila_db` and `./data/ila_students.db`).
  * Configures the local Ollama endpoint (`http://localhost:11434`).
  * Specifies active offline models:
    * LLM: `qwen3:8b` (can be swapped in one line to `phi4-mini` or `llama3.2:3b`).
    * Embeddings: `nomic-embed-text` (768-dimensional dense vector space).
  * Sets RAG hyperparameters:
    * `CHUNK_SIZE = 512` tokens.
    * `CHUNK_OVERLAP = 64` tokens.
    * `TOP_K_RESULTS = 4` passages.
    * `MAX_DISTANCE_THRESHOLD = 1.5` (cosine distance boundary for anti-hallucination).
  * Sets adaptive learning thresholds:
    * `DEFAULT_MASTERY_THRESHOLD = 0.75` (75% probability threshold for topic mastery).
    * `WEAK_TOPIC_RATIO = 0.70` (70% weak-topic practice, 30% strong-topic review).

#### 4. [`core/prompts.py`](file:///c:/Users/rehan/OneDrive/Desktop/MINI%20PROJECT/core/prompts.py)
* **Role**: **Prompt Engineering & Dynamic Pedagogical Persona Injection**.
* **What it does**:
  * `DEFAULT_SYSTEM_PROMPT`: Enforces strict grounding rules — instructs the model to answer *strictly* from the provided context passages and say *"I don't have enough information"* if the topic is absent, prohibiting external hallucinations.
  * `build_adaptive_system_prompt()`: Dynamically compiles a specialized system prompt under 200 tokens based on the student's current profile:
    * **Mastery Level**: Adjusts language between *Beginner* (accessible, jargon-free), *Intermediate* (balanced), and *Advanced* (technical rigor, edge cases).
    * **Pedagogical Style**: Adapts between *Analogies* (real-world metaphors), *Step-by-step* (logical proofs), *Concise* (exam bullets), and *Socratic* (guided hints).
    * **Weak Topics**: Injects the student's struggle concepts so explanations prioritize those areas.
  * `build_rag_user_prompt()`: Wraps retrieved context passages and the student's question into structured sections.
  * `build_quiz_prompt()`: Prompt template forcing the LLM to output pure JSON containing multiple-choice questions, answer options, correct answer keys, and concept tags.

#### 5. [`core/ingest.py`](file:///c:/Users/rehan/OneDrive/Desktop/MINI%20PROJECT/core/ingest.py)
* **Role**: **Document Processing, Vectorization & Deduplication Pipeline**.
* **What it does**:
  * `compute_file_hash()`: Generates a SHA-256 cryptographic digest of uploaded document bytes.
  * `extract_text_from_pdf()`: Uses PyMuPDF (`fitz`) to extract text page-by-page, preserving line integrity and tracking page numbers.
  * `chunk_document_pages()`: Employs `RecursiveCharacterTextSplitter` to create 512-token overlapping chunks tagged with rich metadata (`source_file`, `page_number`, `chunk_index`, `doc_hash`).
  * `generate_embedding()` & `batch_generate_embeddings()`: Sends chunks in batches to the local `nomic-embed-text` model via Ollama REST API, returning 768-dimensional float arrays.
  * `get_vector_store()`: Manages a singleton connection to the persistent ChromaDB collection (`study_material`) configured with cosine distance space (`hnsw:space: cosine`).
  * `ingest_document()`: Master coordinator:
    1. Checks SHA-256 hash against SQLite cache — if already indexed, skips re-embedding immediately (instant response).
    2. Parses PDF page-by-page.
    3. Splits into chunks with deterministic IDs (`{hash}_p{page}_c{index}`).
    4. Computes vector embeddings offline.
    5. Stores text + vectors + metadata into ChromaDB.
    6. Logs document record in SQLite `indexed_documents`.
  * `query_vector_store()`: Embeds queries and performs nearest-neighbor vector search, returning top matching passages with cosine distances and normalized similarity scores.

#### 6. [`core/rag.py`](file:///c:/Users/rehan/OneDrive/Desktop/MINI%20PROJECT/core/rag.py)
* **Role**: **Retrieval-Augmented Generation Query Engine & Streaming Interface**.
* **What it does**:
  * `format_retrieved_context()`: Assembles retrieved ChromaDB chunks into structured passages with headers (`[Passage X | Source: file.pdf (Page Y) | Relevance: Z%]`) and compiles deduplicated source citations.
  * `query_rag_stream()`: **Real-time token streaming generator** designed for web UIs:
    1. Performs semantic vector search on ChromaDB.
    2. Rejects out-of-domain queries if minimum distance exceeds `MAX_DISTANCE_THRESHOLD`.
    3. Builds the adapted system prompt from the student's profile.
    4. Sends a streaming HTTP POST request to Ollama (`stream: True`).
    5. Yields tokens in real-time as NDJSON lines arrive from the GPU, reducing Time to First Token (TTFT) to ~350ms.
  * `query_rag()`: Synchronous helper that returns the complete answer string, structured citations, and grounding status for automated testing and REST endpoints.

---

### The `db/` Package (Relational Data Layer)

#### 7. [`db/database.py`](file:///c:/Users/rehan/OneDrive/Desktop/MINI%20PROJECT/db/database.py)
* **Role**: **SQLite Database Connection Manager, Schema Versioning & CRUD Operations**.
* **What it does**:
  * Manages SQLite connection to `data/ila_students.db` using `sqlite3.Row` factory.
  * `init_db()`: Initializes the relational tables and performance indexes:
    * `schema_version`: Tracks migration schema versions.
    * `indexed_documents`: Stores `(file_hash, file_name, chunk_count, indexed_at)` for deduplication caching.
    * `quiz_results`: Stores individual quiz question attempts with `(student_id, topic, question, options_json, selected_option, correct_answer, is_correct, explanation, timestamp)`.
    * `student_profiles`: Stores per-student state `(student_id, level, explanation_style, mastery_json, updated_at)`.
    * `spaced_repetition`: Stores SM-2 tracking state `(student_id, topic, ease_factor, interval, repetitions, next_review_date, last_reviewed_at)`.
  * **Database Index**:
    * `CREATE INDEX idx_student_topic ON quiz_results(student_id, topic)`: Optimizes Bayesian mastery computation queries from $O(N)$ full table scans down to $O(\log N)$ B-Tree lookups.
  * Helper functions: `record_quiz_result()`, `get_quiz_history()`, `get_or_create_profile()`, `update_student_profile()`, `reset_student_profile()`, `is_document_indexed()`, `record_indexed_document()`.

---

### The `tests/` Package (Automated Verification Suite)

#### 8. [`tests/test_smoke.py`](file:///c:/Users/rehan/OneDrive/Desktop/MINI%20PROJECT/tests/test_smoke.py) (Phase 1 Test)
* **Purpose**: Verifies that the base operating environment, SQLite, and Ollama offline runtime are operational.
* **Test Steps**:
  1. Initializes SQLite and verifies table creation and sample CRUD writes.
  2. Pings Ollama at `http://localhost:11434/api/tags` and verifies installed models (`nomic-embed-text` and `qwen3:8b`).
  3. Tests vector embedding generation with `nomic-embed-text` and asserts vector dimension equals 768.
  4. Tests streaming text generation with `qwen3:8b` and measures execution latency.

#### 9. [`tests/test_ingest.py`](file:///c:/Users/rehan/OneDrive/Desktop/MINI%20PROJECT/tests/test_ingest.py) (Phase 2 Test)
* **Purpose**: Tests end-to-end PDF processing, chunking, caching, and ChromaDB vector persistence.
* **Test Steps**:
  1. Programmatically creates a 2-page academic PDF (`test_physics_notes.pdf` covering Classical Mechanics and Thermodynamics).
  2. Tests PyMuPDF page-by-page extraction and text cleaning.
  3. Tests full document ingestion into persistent ChromaDB.
  4. Tests SHA-256 deduplication cache (verifies re-uploading the same file skips re-embedding).
  5. Executes two semantic search queries against ChromaDB to verify retrieval accuracy:
     * Query: *"What is the law of inertia..."* $\rightarrow$ correctly retrieves Page 1 (Similarity: 78.2%).
     * Query: *"Explain entropy and heat engines..."* $\rightarrow$ correctly retrieves Page 2 (Similarity: 68.4%).

#### 10. [`tests/test_rag.py`](file:///c:/Users/rehan/OneDrive/Desktop/MINI%20PROJECT/tests/test_rag.py) (Phase 3 Test)
* **Purpose**: Tests the RAG query engine, anti-hallucination guardrails, and dynamic persona adaptation.
* **Test Steps**:
  1. Tests real-time streaming RAG on Newton's laws $\rightarrow$ verifies mathematical formula ($F = m \cdot a$) and Page 1 source citation.
  2. Tests synchronous RAG on Thermodynamics $\rightarrow$ verifies entropy explanation and Page 2 source citation.
  3. Tests anti-hallucination filter on an out-of-domain query (*"How to bake cookies"*) $\rightarrow$ verifies system gracefully states it lacks information rather than making up a recipe.
  4. Tests adaptive explanation styling with a beginner student profile $\rightarrow$ verifies the LLM explains entropy using everyday analogies (e.g., messy room metaphor).

---

### The `data/` Layer (Local Storage)

#### 11. [`data/ila_students.db`](file:///c:/Users/rehan/OneDrive/Desktop/MINI%20PROJECT/data/ila_students.db)
* Embedded SQLite database storing student relational records, quiz logs, and SM-2 review schedules.

#### 12. `data/ila_db/`
* Directory containing the persistent ChromaDB database:
  * SQLite metadata index (`chroma.sqlite3`).
  * HNSW binary graph index directories storing 768-dimensional float embeddings on disk.

---

## 4. Architectural Dataflow Diagrams

### A. Document Ingestion Pipeline (Phase 2)

```
[Student Uploads PDF]
         │
         ▼
[Compute SHA-256 Hash] ───(Match Found?)───► [Return Cached Status (Instant)]
         │ (No)
         ▼
[PyMuPDF (fitz)] ───► Page-by-Page Clean Text Extraction
         │
         ▼
[LangChain Splitter] ───► 512-Token Overlapping Chunks (64 overlap)
         │
         ▼
[Ollama nomic-embed-text] ───► Generate 768-dim Vectors (Local GPU)
         │
         ▼
[ChromaDB] ───► Save Vectors, Text, Metadata to ./data/ila_db/
         │
         ▼
[SQLite] ───► Insert into indexed_documents table
```

### B. Grounded RAG Query Pipeline (Phase 3)

```
[Student Asks Question]
         │
         ▼
[Ollama nomic-embed-text] ───► Embed Question Vector (768-dim)
         │
         ▼
[ChromaDB Vector Search] ───► Retrieve Top-4 Chunks (Cosine Distance)
         │
         ├───(Distance > 1.5?)───► [Return "Information Not in Notes" Message]
         │ (Relevant Match)
         ▼
[db/database.py] ───► Fetch Student Profile (Level, Weak Topics, Style)
         │
         ▼
[core/prompts.py] ───► Compile Adaptive Grounded Prompt Template
         │
         ▼
[Ollama qwen3:8b] ───► Real-time NDJSON Token Streaming (Local GPU)
         │
         ▼
[Streamlit / Output] ───► Render Tokens on Screen with Exact Page Citations
```

---

## 5. Upcoming Files (Phases 4, 5 & 6 Roadmap)

| Upcoming File | Target Phase | Planned Responsibility |
| :--- | :--- | :--- |
| `core/quiz.py` | Phase 4 | Prompts LLM for JSON multiple-choice questions from retrieved chunks; validates options and answer keys. |
| `core/adaptive.py` | Phase 4 & 5 | Implements Bayesian Knowledge Tracing ($P(L_t)$ update rules) and the SM-2 Spaced Repetition scheduler. |
| `app.py` | Phase 6 | Multi-page Streamlit application (PDF Upload tab, Chat with Tutor tab, Interactive Quiz tab, Mastery Dashboard). |
| `api/main.py` | Phase 6 | Optional FastAPI REST backend exposing endpoints (`/upload`, `/ask`, `/quiz`, `/progress`) for multi-device LAN usage. |

---

## 6. How to Walk Your Professor Through This in 3 Minutes

When presenting to your professor tomorrow, follow this structured narrative:

1. **The Core Philosophy (30 seconds)**:
   > *"Sir, we are building **ILA (Intelligent Learning Assistant)**. Instead of using cloud APIs that charge fees, hallucinate, and leak student data, ILA is 100% offline. It runs on consumer hardware using local models via Ollama and local vector databases."*

2. **The Codebase Architecture (60 seconds)**:
   > *"We structured our project cleanly following software engineering best practices:*
   > * *Our `core/` package isolates AI logic: `config.py` for parameters, `prompts.py` for versioned prompts, `ingest.py` for PDF chunking and ChromaDB vectorization, and `rag.py` for real-time streaming RAG.*
   > * *Our `db/` package handles persistence: SQLite with custom B-Tree indexing on `(student_id, topic)` for fast lookups.*
   > * *Our `tests/` directory provides test-driven validation for each build phase."*

3. **Current Accomplishments & Live Verification (60 seconds)**:
   > *"So far, we have built and verified Phases 1, 2, and 3:*
   > * *In Phase 1, we validated local 768-dim embeddings and Qwen3 8B streaming.*
   > * *In Phase 2, we built PyMuPDF extraction, 64-token chunk overlap, SHA-256 deduplication caching, and persistent ChromaDB storage.*
   > * *In Phase 3, we built our grounded RAG engine with anti-hallucination guardrails and verified page citations (e.g. Page 1 for Newton's laws, Page 2 for Thermodynamics).*
   > * *We are now proceeding to Phase 4: auto-generating quizzes and modeling student mastery using Bayesian Knowledge Tracing."*

4. **The Live Demo Command**:
   > Run this command in the terminal to demonstrate all tests passing in front of your professor:
   > ```powershell
   > .\ila_env\Scripts\python.exe -u tests\test_rag.py
   > ```
