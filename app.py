# STREAMLIT UI
# Run with:  streamlit run app.py

import streamlit as st
from rag_backend import (
    get_embedding_model,
    get_llm,
    ingest_pdf,
    answer_question,
)

st.set_page_config(page_title="PDF Chatbot", page_icon="📄")
st.title("📄 Chat with your PDF")
st.caption("Upload a PDF and ask questions. Answers come only from the PDF.")


# Load the models only once
@st.cache_resource
def load_models():
    return get_embedding_model(), get_llm()


embedding_model, llm = load_models()


# ---------------------------------------------------
# Sidebar: upload PDF
# ---------------------------------------------------

with st.sidebar:
    st.header("Upload")
    uploaded = st.file_uploader("Choose a PDF", type="pdf")

if not uploaded:
    st.info("👈 Upload a PDF from the sidebar to begin.")
    st.stop()


# ---------------------------------------------------
# Store the PDF in ChromaDB (only when a new file is uploaded)
# ---------------------------------------------------

file_key = (uploaded.name, uploaded.size)

if st.session_state.get("file_key") != file_key:
    with st.spinner("Reading the PDF and storing it in ChromaDB..."):
        st.session_state.vector_db = ingest_pdf(uploaded.getvalue(), embedding_model)
        st.session_state.file_key = file_key
        st.session_state.messages = []

count = st.session_state.vector_db._collection.count()
st.sidebar.success(f"✅ {uploaded.name}\n\n{count} chunks stored")


# ---------------------------------------------------
# Chat
# ---------------------------------------------------

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

question = st.chat_input("Ask a question about the PDF")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the PDF..."):
            try:
                answer, pages = answer_question(
                    st.session_state.vector_db, llm, question
                )
            except Exception as e:
                answer, pages = f"Error: {e}", []

        st.write(answer)
        if pages:
            st.caption("Source pages: " + ", ".join(str(p) for p in pages))

    st.session_state.messages.append({"role": "assistant", "content": answer})
