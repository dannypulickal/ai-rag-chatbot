"""FastAPI application for a RAG-powered chatbot with document management.

This module provides a REST API with the following endpoints:
- POST /chat: Chat with the AI using retrieved documents
- POST /upload-doc: Upload and index documents for retrieval
- GET /documents: List all uploaded documents
- DELETE /documents/{file_id}: Delete a document by ID

All chat interactions are logged to the database and vector store.
"""

from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException

load_dotenv(Path(__file__).with_name(".env"))

from pydantic_models import QueryInput, QueryResponse, DocumentInfo
from langchain_utils import get_rag_chain
from db_utils import insert_application_logs, get_chat_history, get_all_documents, insert_document_record, delete_document_record
from chroma_utils import index_document_to_chroma, delete_doc_from_chroma
import os
import uuid
import logging
import shutil

# Set up logging
logging.basicConfig(filename='app.log', level=logging.INFO)

# Initialize FastAPI app
app = FastAPI()


@app.post("/chat", response_model=QueryResponse)
def chat(query_input: QueryInput):
    """Answer a user question using RAG with chat history.

    Process:
    1. Create or retrieve a session ID
    2. Fetch prior chat history for context
    3. Execute the RAG chain with the question
    4. Log the interaction to the database
    5. Return the answer and session ID

    Args:
        query_input (QueryInput): Request object with:
            - question: The user's query (str)
            - session_id: Optional session identifier (str)
            - model: The AI model to use (enum)

    Returns:
        QueryResponse: Response object with:
            - answer: The AI-generated response (str)
            - session_id: Session identifier for follow-ups (str)
            - model: The model used (enum)
    """
    session_id = query_input.session_id or str(uuid.uuid4())
    logging.info(f"Session ID: {session_id}, User Query: {query_input.question}, Model: {query_input.model.value}")

    chat_history = get_chat_history(session_id)
    rag_chain = get_rag_chain(query_input.model.value)
    answer = rag_chain.invoke({
        "input": query_input.question,
        "chat_history": chat_history
    })['answer']

    insert_application_logs(session_id, query_input.question, answer, query_input.model.value)
    logging.info(f"Session ID: {session_id}, AI Response: {answer}")
    return QueryResponse(answer=answer, session_id=session_id, model=query_input.model)

@app.post("/upload-doc")
def upload_and_index_document(file: UploadFile = File(...)):
    """Upload a document and index it for retrieval.

    Process:
    1. Validate the file extension (pdf, docx, or html)
    2. Save the file temporarily
    3. Create a document record in the database
    4. Index the document in Chroma vector store
    5. Clean up the temporary file

    If indexing fails, rollback the database record.

    Args:
        file (UploadFile): The uploaded file.

    Returns:
        dict: Success response with:
            - message: Confirmation message (str)
            - file_id: Database ID of the indexed document (int)

    Raises:
        HTTPException: 400 if file type is not supported
        HTTPException: 500 if indexing fails
    """
    allowed_extensions = ['.pdf', '.docx', '.html']
    file_extension = os.path.splitext(file.filename)[1].lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Allowed types are: {', '.join(allowed_extensions)}")

    temp_file_path = f"temp_{file.filename}"

    try:
        # Save the uploaded file to a temporary file
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        file_id = insert_document_record(file.filename)
        success = index_document_to_chroma(temp_file_path, file_id)

        if success:
            return {"message": f"File {file.filename} has been successfully uploaded and indexed.", "file_id": file_id}
        else:
            delete_document_record(file_id)
            raise HTTPException(status_code=500, detail=f"Failed to index {file.filename}.")
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.get("/documents", response_model=list[DocumentInfo])
def list_documents():
    """Retrieve metadata for all uploaded documents.

    Returns:
        list[DocumentInfo]: List of documents ordered by upload time (newest first).
    """
    return get_all_documents()

@app.delete("/documents/{file_id}")
def delete_document(file_id: int):
    """Delete a document from both Chroma and the database.

    Process:
    1. Delete the document chunks from Chroma vector store
    2. Delete the document metadata from the database

    Args:
        file_id (int): The database ID of the document to delete.

    Returns:
        dict: Success message confirming deletion.

    Raises:
        HTTPException: 500 if Chroma deletion fails
        HTTPException: 500 if database deletion fails (even after Chroma deletion)
    """
    chroma_deleted = delete_doc_from_chroma(file_id)

    if not chroma_deleted:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete document with ID {file_id} from Chroma.",
        )

    db_deleted = delete_document_record(file_id)

    if not db_deleted:
        raise HTTPException(
            status_code=500,
            detail=f"Deleted from Chroma but failed to delete document with ID {file_id} from the database.",
        )

    return {"message": f"Document with ID {file_id} has been successfully deleted."}
