# 📄 Document-RAG-Chatbot

A PDF-based **Retrieval-Augmented Generation (RAG) chatbot** that allows users to ask questions about the content of a document and receive AI-generated answers based on the retrieved information.

The project combines document processing, vector embeddings, and Google's Gemini LLM to build a document question-answering system.

---

## ✨ Features

- 📄 PDF document processing
- 🧩 Text chunking for efficient retrieval
- 🧠 Google Gemini embeddings
- 🗄️ ChromaDB vector database
- 🤖 Gemini-powered question answering
- 💬 Interactive Streamlit chatbot interface

---

## 🛠️ Tech Stack

- 🐍 **Python**
- 🎈 **Streamlit** – User interface
- 🦜 **LangChain** – RAG pipeline and document processing
- 🤖 **Google Gemini** – LLM and embeddings
- 🗄️ **ChromaDB** – Vector database
- 📑 **PyPDF** – PDF document loading
- 🔐 **python-dotenv** – Environment variable management

---

## 🧠 How It Works

```text
📄 PDF Document
       ↓
📑 Text Extraction
       ↓
✂️ Text Chunking
       ↓
🧠 Generate Embeddings
       ↓
🗄️ Store in ChromaDB
       ↓
❓ User Question
       ↓
🔍 Similarity Search
       ↓
📚 Retrieve Relevant Context
       ↓
🤖 Gemini LLM
       ↓
💬 Generated Answer
 ```
## 📂 Folder Structure

```text
Document-RAG-Chatbot/
├── app.py
├── create_chroma.py
├── embedding.py
├── main.py
├── rag_backend.py
├── requirements.txt
├── .gitignore
└── README.md
```
## 🖥️ Demo Screenshots

## screenshot 1

<img width="1920" height="1020" alt="Screenshot 2026-09-24 222542" src="https://github.com/user-attachments/assets/02daf260-6f3e-4edb-b445-b040813a72c2" />

## screenshot 2

<img width="1920" height="1020" alt="Screenshot 2026-09-24 222651" src="https://github.com/user-attachments/assets/06cabba0-0374-4c37-80d7-522ab1daf568" />

## screenshot 3

<img width="1920" height="1020" alt="Screenshot 2026-09-24 222717" src="https://github.com/user-attachments/assets/a9f25ef4-f511-4643-9e3e-b9c8b441fb9c" />

## screenshot 4

<img width="1920" height="1020" alt="Screenshot 2026-09-24 222741" src="https://github.com/user-attachments/assets/6900e93d-6e0e-42f2-926c-4b587ea7e152" />




