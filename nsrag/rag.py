"""
nsrag/rag.py (updated)

- Full RAG with sentence-transformers embeddings stored locally
- Uses Ollama for generation (local: ollama serve)
- Uses DuckDuckGo free HTML search for web results (no API key required)
- Returns line-level PDF citations (filename, page, start_line-end_line)
- Persistent vector store under nsrag/vstore/
- Requirements: sentence-transformers, numpy, PyPDF2, requests, beautifulsoup4, langchain-community
"""

import os
import json
import pathlib
import hashlib
import numpy as np
from typing import List, Dict, Tuple, Optional
import re
import requests
from io import BytesIO

# PDF reading
import PyPDF2

# HTML parsing for DuckDuckGo
from bs4 import BeautifulSoup

# Embeddings model (sentence-transformers)
try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    raise RuntimeError("Missing dependency: sentence-transformers. Install with `pip install sentence-transformers`") from e

# Ollama LLM wrapper via langchain_community
try:
    from langchain_community.llms import Ollama
except Exception as e:
    raise RuntimeError("Missing dependency: langchain-community (for Ollama). Install accordingly.") from e

# Globals / config
DATA_DIR = pathlib.Path(__file__).parent / "data"
VSTORE_DIR = pathlib.Path(__file__).parent / "vstore"
VSTORE_DIR.mkdir(exist_ok=True)
EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2")  # sentence-transformers model name
EMBED_BATCH = 32
TOP_K = 5
MIN_SIMILARITY = 0.30  # threshold to consider PDF evidence reliable
PERSIST_META = VSTORE_DIR / "metadata.json"
PERSIST_EMB = VSTORE_DIR / "embeddings.npy"

# Lazy globals
_embed_model = None
_ollama_model = None

def get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    return _embed_model

def get_ollama_model():
    global _ollama_model
    if _ollama_model is None:
        model_name = os.environ.get("OLLAMA_MODEL", "llama3.2:latest")
        _ollama_model = Ollama(model=model_name)
    return _ollama_model

# Utilities
def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def chunk_text_lines(text: str, max_chars: int = 800) -> List[Tuple[int,int,str]]:
    """
    Break page text into chunks of up to max_chars characters, preserving line boundaries.
    Returns list of tuples (start_line_idx, end_line_idx, chunk_text) with 1-based line numbers.
    """
    lines = [re.sub(r'\s+', ' ', ln).strip() for ln in text.splitlines() if ln.strip()]
    chunks = []
    i = 0
    while i < len(lines):
        j = i + 1
        while j < len(lines) and len(" ".join(lines[i:j+1])) <= max_chars:
            j += 1
        chunk = " ".join(lines[i:j])
        chunks.append((i+1, j, chunk))
        i = j
    return chunks

def load_pdfs_to_documents(pdf_dir: pathlib.Path = DATA_DIR) -> List[Dict]:
    """
    Read all PDFs from pdf_dir and produce document chunks with metadata:
    { 'id', 'filename', 'page', 'start_line', 'end_line', 'text' }
    """
    docs = []
    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        try:
            with open(pdf_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page_idx, page in enumerate(reader.pages):
                    raw = page.extract_text() or ""
                    chunks = chunk_text_lines(raw)
                    for start, end, chunk_text in chunks:
                        doc_id = hashlib.sha1(f"{pdf_path.name}-{page_idx+1}-{start}-{end}".encode()).hexdigest()
                        docs.append({
                            "id": doc_id,
                            "filename": pdf_path.name,
                            "page": page_idx + 1,
                            "start_line": start,
                            "end_line": end,
                            "text": chunk_text
                        })
        except Exception as e:
            print(f"[rag] Failed to read {pdf_path}: {e}")
            continue
    return docs

def build_vectorstore(force: bool = False) -> Dict:
    """
    Build embeddings for PDF chunks and persist them.
    """
    if PERSIST_META.exists() and PERSIST_EMB.exists() and not force:
        meta = json.loads(PERSIST_META.read_text(encoding="utf-8"))
        emb = np.load(str(PERSIST_EMB))
        return {"meta": meta, "emb": emb}
    docs = load_pdfs_to_documents(DATA_DIR)
    if not docs:
        meta = {"entries": []}
        np.save(str(PERSIST_EMB), np.zeros((0, 768)))
        PERSIST_META.write_text(json.dumps(meta), encoding="utf-8")
        return {"meta": meta, "emb": np.zeros((0,768))}
    texts = [d["text"] for d in docs]
    model = get_embed_model()
    embeddings = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i:i+EMBED_BATCH]
        emb_batch = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        embeddings.append(emb_batch)
    emb_array = np.vstack(embeddings)
    meta = {"entries": docs}
    PERSIST_META.write_text(json.dumps(meta), encoding="utf-8")
    np.save(str(PERSIST_EMB), emb_array)
    return {"meta": meta, "emb": emb_array}

def load_vectorstore() -> Dict:
    if PERSIST_META.exists() and PERSIST_EMB.exists():
        meta = json.loads(PERSIST_META.read_text(encoding="utf-8"))
        emb = np.load(str(PERSIST_EMB))
        return {"meta": meta, "emb": emb}
    else:
        return build_vectorstore()

def semantic_search(query: str, top_k: int = TOP_K) -> List[Tuple[Dict, float]]:
    vs = load_vectorstore()
    meta = vs["meta"]["entries"]
    emb = vs["emb"]
    if len(meta) == 0 or emb.shape[0] == 0:
        return []
    model = get_embed_model()
    q_emb = model.encode([query], convert_to_numpy=True)[0]
    sims = np.dot(emb, q_emb) / (np.linalg.norm(emb, axis=1) * (np.linalg.norm(q_emb)+1e-12))
    sims = np.nan_to_num(sims)
    idx_sorted = np.argsort(-sims)[:top_k]
    results = []
    for idx in idx_sorted:
        results.append((meta[idx], float(sims[idx])))
    return results

# Free web search using DuckDuckGo HTML (no API key)
def duckduckgo_search(query: str, num_results: int = 3) -> List[Tuple[str, str, str]]:
    """
    Returns list of tuples (title, url, snippet)
    Uses the public DuckDuckGo HTML endpoint, parses result items.
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; QuizBot/1.0; +https://example.com)"}
        resp = requests.post("https://duckduckgo.com/html/", data={"q": query}, headers=headers, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        results = []
        # DuckDuckGo HTML uses result classes; try multiple selectors for robustness
        for r in soup.select(".result")[:num_results]:
            a = r.select_one("a.result__a") or r.select_one("a")
            title = a.get_text(strip=True) if a else ""
            link = a.get("href") if a else ""
            snippet_el = r.select_one(".result__snippet") or r.select_one(".snippet") or r.select_one("a")
            snippet = snippet_el.get_text(strip=True) if snippet_el else ""
            if link and link.startswith("/l/?kh="):
                # DuckDuckGo sometimes returns redirect links; try to extract actual URL parameter 'uddg'
                from urllib.parse import parse_qs, urlparse, unquote
                qs = urlparse(link).query
                parsed = parse_qs(qs)
                if "uddg" in parsed:
                    link = unquote(parsed["uddg"][0])
            results.append((title, link, snippet))
        # Fallback: try another selector
        if not results:
            for r in soup.select(".result__body")[:num_results]:
                a = r.select_one("a.result__a") or r.select_one("a")
                title = a.get_text(strip=True) if a else ""
                link = a.get("href") if a else ""
                snippet_el = r.select_one(".result__snippet")
                snippet = snippet_el.get_text(strip=True) if snippet_el else ""
                results.append((title, link, snippet))
        return results[:num_results]
    except Exception as e:
        print(f"[duckduckgo] search failed: {e}")
        return []

def build_prompt(question: str, retrieved: List[Tuple[Dict,float]], web_results: List[Tuple[str,str,str]]) -> str:
    pieces = []
    if retrieved:
        pieces.append("PDF evidence:")
        for doc, score in retrieved:
            txt = doc["text"]
            pieces.append(f"---\nFile: {doc['filename']}\nPage: {doc['page']}\nLines: {doc['start_line']}-{doc['end_line']}\nSnippet: {txt}\n---\nSimilarity: {score:.3f}")
    if web_results:
        pieces.append("Web evidence:")
        for title, url, snippet in web_results:
            pieces.append(f"- {title} | {url}\n{snippet}")
    context = "\n\n".join(pieces) if pieces else "No supporting documents found."
    prompt = f"""You are a helpful and precise network-security assistant. Answer the question using the provided evidence. 
If the answer is supported by the PDF snippets, explicitly cite them inline using the format [filename.pdf page X lines Y-Z]. 
If the answer uses a web source, cite the URL inline in square brackets. 
Be concise and provide step-by-step only if asked.

Question:
{question}

Context:
{context}

If you cannot find a confident answer in the provided evidence, say so and offer suggestions for where to look next.
"""
    return prompt

def answer_question(question: str, use_web: bool = False, top_k: int = TOP_K) -> Tuple[str, List[str]]:
    """
    Perform semantic search over indexed PDF chunks, optionally web lookup via DuckDuckGo, then call Ollama to answer.
    Returns (answer_text, sources_list)
    """
    # ensure vectorstore built
    vs = load_vectorstore()
    retrieved = semantic_search(question, top_k=top_k)
    sources = []
    web_results = []
    strong = [r for r in retrieved if r[1] >= MIN_SIMILARITY]
    if not strong and use_web:
        web_results = duckduckgo_search(question, num_results=5)
    prompt = build_prompt(question, retrieved, web_results)
    model = get_ollama_model()
    try:
        resp = model.invoke(prompt)
    except Exception as e:
        raise RuntimeError(f"Ollama invocation failed: {e}")
    # Build sources list: PDF entries if available and above threshold; otherwise web urls
    if retrieved:
        for doc, score in retrieved:
            if score >= MIN_SIMILARITY:
                sources.append(f"{doc['filename']} (page {doc['page']}, lines {doc['start_line']}-{doc['end_line']})")
    if not sources and web_results:
        sources = [url for (_t, url, _s) in web_results if url]
    return resp, sources

# CLI helpers (for testing)
if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--rebuild', action='store_true')
    p.add_argument('--question', type=str, default=None)
    args = p.parse_args()
    if args.rebuild:
        print('[rag] Building vectorstore...')
        build_vectorstore(force=True)
        print('[rag] Done.')
    if args.question:
        a, s = answer_question(args.question, use_web=True)
        print('Answer:', a)
        print('Sources:', s)
