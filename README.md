# AI RAG Chatbot

A local Retrieval-Augmented Generation (RAG) chatbot built with FastAPI,
Streamlit, LangChain, OpenAI, Chroma, and SQLite.

The app lets you upload documents, index them into a local Chroma vector store,
and ask questions against those documents through a Streamlit chat interface.
FastAPI handles document upload, retrieval, chat responses, and chat history.

## Requirements

- Python 3.12+
- Poetry
- An OpenAI API key

## Setup

Install dependencies from the project root:

```bash
cd to project directory (ai-rag-chatbot)
poetry install
```

Create an `.env` file for the backend:

```bash
touch src/ai_rag_chatbot/api/.env
```

Add your OpenAI API key:

```bash
OPENAI_API_KEY=your_openai_api_key
```

## Run Locally

Start the FastAPI backend:

```bash
cd /project_directory/src/ai_rag_chatbot/api
poetry run uvicorn main:app --reload
```

The backend runs at:

```text
http://127.0.0.1:8000
```

In a second terminal, start the Streamlit frontend:

```bash
cd /project_directory/src/ai_rag_chatbot/app
poetry run streamlit run streamlit_app.py
```

Open the Streamlit URL shown in the terminal, usually:

```text
http://localhost:8501
```

## Using The App

1. Select an OpenAI model from the sidebar.
2. Upload a `.pdf`, `.docx`, or `.html` document.
3. Ask questions in the chat box.
4. Refresh or delete uploaded documents from the sidebar.

## Configuration

By default, the frontend calls the backend at `http://localhost:8000`.
To point the frontend at another backend URL, set:

```bash
export RAG_API_BASE_URL="https://your-api.example.com"
```

Then run Streamlit as usual.

You can also configure this with Streamlit secrets:

```toml
api_base_url = "https://your-api.example.com"
```

## Local Data

The app creates local files while running:

- `rag_app.db` stores chat logs and uploaded document metadata.
- `chroma_db/` stores local vector embeddings.
