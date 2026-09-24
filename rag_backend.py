import os
import hashlib
import tempfile

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma


# ---------------------------------------------------
# Settings
# ---------------------------------------------------

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")
CHROMA_DIR = "./chroma_db"
EMBEDDING_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-3.6-flash"
NOT_FOUND = "This information is not present in the PDF."


# ---------------------------------------------------
# Models
# ---------------------------------------------------

def get_embedding_model():
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=API_KEY
    )


def get_llm():
    return ChatGoogleGenerativeAI(
        model=CHAT_MODEL,
        google_api_key=API_KEY,
        temperature=0
    )


# ---------------------------------------------------
# Ingestion: PDF -> chunks -> embeddings -> ChromaDB
# ---------------------------------------------------

def ingest_pdf(pdf_bytes, embedding_model):
    """Store an uploaded PDF in ChromaDB and return the vector store.

    Each PDF gets its own collection (named from the file's hash),
    so different PDFs never mix, and the same PDF is not stored twice.
    """
    file_id = hashlib.md5(pdf_bytes).hexdigest()

    vector_db = Chroma(
        collection_name="pdf_" + file_id,
        embedding_function=embedding_model,
        persist_directory=CHROMA_DIR
    )

    # Already stored earlier? Then skip the slow part.
    if vector_db._collection.count() > 0:
        return vector_db

    # Save upload to a temporary file (PyPDFLoader needs a path)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        temp_path = tmp.name

    try:
        docs = PyPDFLoader(temp_path).load()
    finally:
        os.remove(temp_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(docs)

    vector_db.add_documents(chunks)
    return vector_db


# ---------------------------------------------------
# Question answering: ChromaDB -> Gemini
# ---------------------------------------------------

def _to_text(content):
    """Gemini sometimes returns a list of parts instead of a plain string."""
    if isinstance(content, str):
        return content
    parts = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, dict) and "text" in part:
            parts.append(part["text"])
    return "".join(parts)


def answer_question(vector_db, llm, question, k=4):
    """Return (answer, pages_used). Answers only from the PDF."""
    results = vector_db.similarity_search(question, k=k)

    if not results:
        return NOT_FOUND, []

    context = "\n\n".join(doc.page_content for doc in results)

    prompt = f"""You are a helpful assistant answering questions about a PDF.
Use ONLY the context below to answer.
If the answer is not in the context, reply exactly:
"{NOT_FOUND}"

Context:
{context}

Question: {question}

Answer:"""

    response = llm.invoke(prompt)
    answer = _to_text(response.content).strip()

    if NOT_FOUND.lower() in answer.lower():
        return NOT_FOUND, []

    pages = sorted({doc.metadata.get("page", 0) + 1 for doc in results})
    return answer, pages
