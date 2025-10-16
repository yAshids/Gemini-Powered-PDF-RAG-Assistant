import streamlit as st
from io import BytesIO
from rag import index_pdf_and_text, PDFRAG

st.set_page_config(page_title="PDF RAG with Gemini", page_icon="🧭", layout="wide")
st.title("🧭 PDF RAG (Gemini)")
st.caption("Upload a PDF or paste text, index chunks, then ask grounded questions powered by Gemini.")

# Session state
if "rag" not in st.session_state:
    st.session_state.rag: PDFRAG | None = None

left, right = st.columns([1,1], vertical_alignment="top")

with left:
    st.subheader("Index data")
    pdf = st.file_uploader("Upload PDF", type=["pdf"])
    pasted = st.text_area("Or paste text", height=160, placeholder="Optional extra content")
    csize = st.number_input("Chunk size", 100, 1200, 500, 50)
    ovlap = st.number_input("Overlap", 0, 400, 50, 10)

    if st.button("Build index", use_container_width=True):
        try:
            st.session_state.rag = index_pdf_and_text(pdf, pasted, int(csize), int(ovlap))
            st.success(f"Indexed {len(st.session_state.rag.chunks)} chunks.")
        except Exception as e:
            st.session_state.rag = None
            st.error(f"Indexing failed: {e}")

    if st.button("Clear index", use_container_width=True):
        st.session_state.rag = None
        st.success("Cleared index.")

with right:
    st.subheader("Ask a question")
    query = st.text_input("Question", placeholder="e.g., What are the refund terms in section 2.1?")
    topk = st.slider("Top-K retrieval", 1, 20, 5)
    temperature = st.slider("Creativity (Gemini temperature)", 0.0, 1.0, 0.2, 0.05)
    # Note: Temperature slider shown for UI; the prompt is strict. Adjust model config in rag.py if needed.

    if st.button("Retrieve + Answer", use_container_width=True):
        if not st.session_state.rag:
            st.warning("Please build the index first.")
        elif not query.strip():
            st.warning("Enter a question.")
        else:
            try:
                hits = st.session_state.rag.search(query, top_k=int(topk))
                if not hits:
                    st.info("No results found. Increase Top-K or adjust chunking.")
                else:
                    contexts = [c for c, _score in hits]
                    answer = st.session_state.rag.generate_answer(query, contexts)
                    st.markdown("### Answer")
                    st.write(answer or "No answer returned.")
                    with st.expander("Retrieved contexts"):
                        for i, (c, s) in enumerate(hits, 1):
                            st.markdown(f"**Chunk {i} (score {s:.3f})**")
                            st.write(c)
            except Exception as e:
                st.error(f"Query failed: {e}")
