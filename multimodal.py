import io
import pdfplumber

def extract_text_from_upload(uploaded_file):
    """
    Works with Streamlit's UploadedFile buffer.
    Supports .pdf and .txt.
    """
    name = (uploaded_file.name or "").lower()
    raw = uploaded_file.read()
    try:
        if name.endswith(".txt"):
            return raw.decode("utf-8", errors="ignore")
        if name.endswith(".pdf"):
            with pdfplumber.open(io.BytesIO(raw)) as pdf:
                parts = []
                for p in pdf.pages:
                    parts.append(p.extract_text() or "")
            return "\n".join(parts)
        raise ValueError("Unsupported file type. Use PDF or TXT.")
    finally:
        try:
            uploaded_file.seek(0)
        except Exception:
            pass
