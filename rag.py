import os
import re
from io import BytesIO
from typing import List, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY missing in .env")

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
# Choose a stable, current model name; adjust if needed
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL", "models/gemini-1.5-flash")
gen_model = genai.GenerativeModel(GEMINI_MODEL_NAME)

# Embedding model
EMBED_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

class PDFRAG:
    def __init__(self):
        self.embedder = SentenceTransformer(EMBED_MODEL_NAME)
        self.index = None
        self.chunks: List[str] = []
        self._emb_matrix = None  # (n, d) float32, normalized

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        return re.sub(r"\s+", " ", text or "").strip()

    @staticmethod
    def extract_pdf_text(file_bytes: BytesIO) -> str:
        reader = PdfReader(file_bytes)
        pages = [(p.extract_text() or "") for p in reader.pages]
        text = "\n".join(pages)
        return PDFRAG._normalize_whitespace(text)

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        toks = text.split()
        chunks, i = [], 0
        step = max(1, chunk_size - overlap)
        while i < len(toks):
            part = toks[i:i + chunk_size]
            if not part:
                break
            chunks.append(" ".join(part))
            i += step
        return chunks

    def build_index(self, chunks: List[str]) -> None:
        if not chunks:
            raise ValueError("No chunks to index.")
        # Encode and normalize for cosine via inner product
        emb = self.embedder.encode(chunks, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        d = emb.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(emb)
        self.index = index
        self.chunks = chunks
        self._emb_matrix = emb

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        if not self.index or not self.chunks:
            return []
        q = self.embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        D, I = self.index.search(q, top_k)
        I = I[0].tolist()
        D = D[0].tolist()
        results = [(self.chunks[i], D[idx]) for idx, i in enumerate(I) if i != -1]
        return results

    @staticmethod
    def make_prompt(query: str, contexts: List[str], max_chars: int = 3500) -> str:
        ctx = "\n\n---\n\n".join(contexts)[:max_chars]
        return (
            "You are a careful assistant that answers only using the provided context.\n"
            "Instructions:\n"
            "- Use only the context below. Do not use outside knowledge.\n"
            "- If the answer is not present, reply exactly: Not found in context.\n"
            "- Keep the answer concise and cite relevant quotes when helpful.\n\n"
            f"Context:\n{ctx}\n\n"
            f"Question: {query}\n"
            "Answer:"
        )

    def generate_answer(self, query: str, contexts: List[str]) -> str:
        prompt = self.make_prompt(query, contexts)
        resp = gen_model.generate_content(prompt)
        # google-generativeai returns text in .text
        return (resp.text or "").strip()

# Convenience function used by Streamlit app
def index_pdf_and_text(pdf_file: BytesIO, pasted_text: str, chunk_size: int, overlap: int) -> PDFRAG:
    rag = PDFRAG()
    text = ""
    if pdf_file is not None:
        text = rag.extract_pdf_text(pdf_file)
    if pasted_text and pasted_text.strip():
        text = (text + "\n" + pasted_text).strip() if text else pasted_text.strip()
    if not text:
        raise ValueError("No content provided (PDF or pasted text).")
    chunks = rag.chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        raise ValueError("No chunks created. Try smaller chunk size or different document.")
    rag.build_index(chunks)
    return rag
