# 📄 Gemini-Powered PDF RAG Assistant

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline that allows users to query information directly from PDFs using **Google Gemini** for grounded, context-aware responses.

## 🧠 Features
- **PDF ingestion** and text extraction
- **Chunking with overlap** for better context
- **Semantic embeddings** via `SentenceTransformers`
- **Top-K retrieval** using `FAISS` vector store
- **Low-temperature Gemini generation** for precise, factual answers

## ⚙️ Tech Stack
- Python
- SentenceTransformers
- FAISS
- Google Gemini API
- PyPDF2 / pdfplumber

## 🚀 How to Run
```bash
git clone https://github.com/<your-username>/Gemini_PDF_RAG_Assistant.git
cd Gemini_PDF_RAG_Assistant
pip install -r requirements.txt
python app.py
