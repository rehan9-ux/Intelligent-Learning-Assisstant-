# Intelligent Learning Assistant (ILA) 🎓
### An Offline, Privacy-Preserving Adaptive AI Study Assistant

**Team Members:**  
- Krish Shandilya 
- Rehan Khan 

---

## 📌 Overview

**ILA** is a completely self-contained, offline AI tutoring system designed to run on consumer hardware with zero cloud dependencies, zero API costs, and 100% data privacy.

The system unifies three core computer science paradigms:
1. **Retrieval-Augmented Generation (RAG):** Local vector embeddings with ChromaDB to ground answers directly in uploaded PDFs, lecture notes, and textbooks with exact page citations.
2. **Offline Generative AI:** High-precision open-weights LLM (`qwen3:8b`) served locally via Ollama with zero latency to external cloud servers.
3. **Cognitive Machine Learning:** Mathematical modeling of student learning using **Bayesian Knowledge Tracing (BKT)** and **SM-2 Spaced Repetition** to deliver adaptive quizzes and personalize explanation depths.

---

## 🏗️ Architecture & Project Structure

```text
MINI PROJECT/
│
├── core/                                    # AI, Ingestion & RAG Logic
│   ├── config.py                            # Central configurations (paths, models, thresholds)
│   ├── prompts.py                           # System prompts & adaptive templates
│   ├── ingest.py                            # PDF extraction, chunking & ChromaDB vector indexing
│   └── rag.py                               # Query retrieval, context grounding & streaming answers
│
├── db/                                      # Relational Database Layer
│   └── database.py                          # SQLite schemas, B-tree indexes & student metrics
│
├── tests/                                   # Automated Test Suite
│   ├── test_smoke.py                        # Offline runtime & Ollama connectivity test
│   ├── test_ingest.py                       # Document parsing & chunking verification
│   └── test_rag.py                          # Vector similarity & RAG pipeline verification
│
├── data/                                    # Local Persistent Storage (.gitignore'd)
│   ├── ila_students.db                      # Local SQLite database (profiles, quiz logs)
│   └── ila_db/                              # ChromaDB persistent vector database
│
├── ask.py                                   # Interactive Terminal CLI for PDF Q&A
├── requirements.txt                         # Pinned Python package dependencies
├── PROJECT_STRUCTURE_AND_CODEBASE_GUIDE.md  # Detailed code walkthrough & architecture guide
└── ILA_STUDY_GUIDE_AND_VIVA_PREP.md         # Comprehensive viva prep & theory guide
```

---

## 🚀 Getting Started (For Team Members)

### 1. Prerequisites
- **Python 3.10+** (tested on 3.11/3.14)
- **Ollama**: [Download Ollama for Windows](https://ollama.com/download)

Pull the required offline models:
```bash
ollama pull nomic-embed-text
ollama pull qwen3:8b
```

### 2. Setup Virtual Environment
Clone the repository and create your local virtual environment:
```bash
# Clone the repository
git clone <YOUR_GITHUB_REPO_URL>
cd "MINI PROJECT"

# Create virtual environment
python -m venv ila_env

# Activate virtual environment
# Windows PowerShell:
.\ila_env\Scripts\Activate.ps1
# Windows Command Prompt:
.\ila_env\Scripts\activate.bat
# Linux/macOS:
source ila_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Verify Setup
Run the smoke and RAG tests to verify everything is working locally:
```bash
python tests/test_smoke.py
python tests/test_rag.py
```

### 4. Interactive CLI Q&A
Index any PDF document and ask questions from it directly:
```bash
python ask.py
# Or supply a PDF path directly:
python ask.py "sample.pdf"
```

---

## 🛡️ License & Academic Integrity
Developed as part of the B.Tech Computer Science & Engineering curriculum.
All rights reserved © 2026 Krish Shandilya & Rehan Khan.
