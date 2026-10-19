import os
import hashlib
import requests
import fitz  # PyMuPDF
import chromadb
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Callable, Union
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.config import (
    CHROMA_DB_DIR,
    OLLAMA_BASE_URL,
    OLLAMA_EMBED_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K_RESULTS
)
from db.database import (
    is_document_indexed,
    record_indexed_document,
    get_indexed_documents
)

# Initialize ChromaDB persistent client singleton
_chroma_client: Optional[chromadb.PersistentClient] = None
_collection: Optional[chromadb.Collection] = None


def get_vector_store() -> chromadb.Collection:
    """Initializes and returns the persistent ChromaDB collection for study materials."""
    global _chroma_client, _collection
    if _collection is None:
        CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
        # Use cosine distance for similarity
        _collection = _chroma_client.get_or_create_collection(
            name="study_material",
            metadata={"hnsw:space": "cosine"}
        )
    return _collection


def compute_file_hash(content_bytes: bytes) -> str:
    """Calculates SHA-256 hash of file bytes for deduplication caching."""
    hasher = hashlib.sha256()
    hasher.update(content_bytes)
    return hasher.hexdigest()


def extract_text_from_pdf(pdf_input: Union[str, Path, bytes], filename: str = "document.pdf") -> List[Dict[str, Any]]:
    """
    Extracts text page-by-page from a PDF using PyMuPDF (fitz).
    Returns a list of dicts: [{'page_number': int, 'text': str, 'total_pages': int, 'source_file': str}]
    """
    if isinstance(pdf_input, (str, Path)):
        doc = fitz.open(str(pdf_input))
    else:
        doc = fitz.open(stream=pdf_input, filetype="pdf")

    total_pages = len(doc)
    pages_data = []

    for page_idx in range(total_pages):
        page = doc.load_page(page_idx)
        text = page.get_text("text").strip()
        
        # Clean up excessive blank lines and formatting artifacts
        cleaned_text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])
        
        if cleaned_text:
            pages_data.append({
                "source_file": filename,
                "page_number": page_idx + 1,
                "total_pages": total_pages,
                "text": cleaned_text
            })

    doc.close()
    return pages_data


def chunk_document_pages(
    pages_data: List[Dict[str, Any]],
    doc_hash: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP
) -> List[Dict[str, Any]]:
    """
    Splits page texts into overlapping chunks with rich metadata tags.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = []
    chunk_counter = 0

    for page_info in pages_data:
        page_text = page_info["text"]
        page_chunks = splitter.split_text(page_text)

        for p_idx, text_chunk in enumerate(page_chunks):
            if not text_chunk.strip():
                continue
            
            chunk_id = f"{doc_hash[:8]}_p{page_info['page_number']}_c{chunk_counter}"
            chunks.append({
                "id": chunk_id,
                "text": text_chunk,
                "metadata": {
                    "source_file": page_info["source_file"],
                    "page_number": page_info["page_number"],
                    "total_pages": page_info["total_pages"],
                    "chunk_index": chunk_counter,
                    "doc_hash": doc_hash
                }
            })
            chunk_counter += 1

    return chunks


def generate_embedding(text: str) -> List[float]:
    """Generates dense embedding vector using Ollama offline nomic-embed-text."""
    url = f"{OLLAMA_BASE_URL}/api/embeddings"
    payload = {
        "model": OLLAMA_EMBED_MODEL,
        "prompt": text
    }
    resp = requests.post(url, json=payload, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"Embedding failed with status {resp.status_code}: {resp.text}")
    return resp.json()["embedding"]


def batch_generate_embeddings(
    texts: List[str],
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> List[List[float]]:
    """Generates embeddings for a batch of texts with progress updates."""
    embeddings = []
    total = len(texts)
    for idx, text in enumerate(texts):
        emb = generate_embedding(text)
        embeddings.append(emb)
        if progress_callback:
            progress_callback(idx + 1, total, f"Vectorizing chunk {idx + 1}/{total}...")
    return embeddings


def ingest_document(
    pdf_input: Union[str, Path, bytes],
    filename: str,
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> Dict[str, Any]:
    """
    Complete ingestion pipeline:
    1. Check SHA-256 hash against SQLite cache to prevent duplicate indexing.
    2. Extract text page-by-page.
    3. Split into overlapping semantic chunks.
    4. Compute embeddings via local Ollama.
    5. Save persistently to ChromaDB.
    6. Record in SQLite indexed_documents.
    """
    # Read bytes for hashing
    if isinstance(pdf_input, (str, Path)):
        with open(str(pdf_input), "rb") as f:
            content_bytes = f.read()
    else:
        content_bytes = pdf_input

    file_hash = compute_file_hash(content_bytes)

    # 1. Deduplication Cache Check
    if is_document_indexed(file_hash):
        if progress_callback:
            progress_callback(100, 100, "Document already indexed (cached).")
        return {
            "status": "cached",
            "file_hash": file_hash,
            "filename": filename,
            "message": f"'{filename}' was already indexed previously. Skipped re-indexing."
        }

    # 2. Extract Text Page-by-Page
    if progress_callback:
        progress_callback(5, 100, f"Extracting text from {filename}...")
    pages_data = extract_text_from_pdf(content_bytes, filename=filename)

    if not pages_data:
        raise ValueError(f"No extractable text found in '{filename}'. It might be a scanned image or empty PDF.")

    # 3. Chunking
    if progress_callback:
        progress_callback(15, 100, f"Splitting text into overlapping chunks...")
    chunks = chunk_document_pages(pages_data, doc_hash=file_hash)

    if not chunks:
        raise ValueError("Document yielded 0 chunks after text splitting.")

    # 4. Generate Embeddings
    chunk_texts = [c["text"] for c in chunks]
    chunk_ids = [c["id"] for c in chunks]
    chunk_metas = [c["metadata"] for c in chunks]

    if progress_callback:
        progress_callback(25, 100, f"Generating local embeddings for {len(chunks)} chunks...")

    def sub_progress(curr: int, total: int, msg: str):
        pct = 25 + int((curr / total) * 65)  # Range 25% -> 90%
        if progress_callback:
            progress_callback(pct, 100, msg)

    embeddings = batch_generate_embeddings(chunk_texts, progress_callback=sub_progress)

    # 5. Store in Persistent ChromaDB
    if progress_callback:
        progress_callback(92, 100, "Saving vectors into ChromaDB...")
    collection = get_vector_store()
    collection.add(
        ids=chunk_ids,
        documents=chunk_texts,
        embeddings=embeddings,
        metadatas=chunk_metas
    )

    # 6. Record in SQLite
    record_indexed_document(file_hash=file_hash, file_name=filename, chunk_count=len(chunks))

    if progress_callback:
        progress_callback(100, 100, f"Successfully indexed {len(chunks)} chunks!")

    return {
        "status": "success",
        "file_hash": file_hash,
        "filename": filename,
        "total_pages": len(pages_data),
        "chunk_count": len(chunks),
        "message": f"Successfully indexed {len(chunks)} chunks across {len(pages_data)} pages from '{filename}'."
    }


def query_vector_store(query_text: str, top_k: int = TOP_K_RESULTS) -> List[Dict[str, Any]]:
    """
    Performs semantic search against ChromaDB using query embeddings.
    Returns list of matching chunk dicts sorted by relevance.
    """
    clean_query = query_text.strip().lower()
    if not clean_query:
        return []

    collection = get_vector_store()
    
    # Check if collection is empty
    if collection.count() == 0:
        return []

    # Generate query vector
    query_vector = generate_embedding(clean_query)

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"]
    )

    formatted_results = []
    if results and "documents" in results and results["documents"]:
        docs = results["documents"][0]
        metas = results["metadatas"][0] if "metadatas" in results and results["metadatas"] else [{}] * len(docs)
        dists = results["distances"][0] if "distances" in results and results["distances"] else [0.0] * len(docs)

        for doc, meta, dist in zip(docs, metas, dists):
            # In cosine space, cosine similarity = 1 - distance
            similarity_score = max(0.0, 1.0 - dist)
            formatted_results.append({
                "text": doc,
                "metadata": meta,
                "distance": dist,
                "similarity_score": round(similarity_score, 4)
            })

    return formatted_results


def get_indexed_stats() -> Dict[str, Any]:
    """Returns overview statistics of indexed documents and chunks."""
    collection = get_vector_store()
    docs = get_indexed_documents()
    return {
        "total_chunks_in_chroma": collection.count(),
        "total_documents": len(docs),
        "documents": docs
    }
