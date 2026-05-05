"""HTTP client utilities for calling the RAG chatbot API.

Provides wrapper functions for the Streamlit frontend to interact
with the FastAPI backend, including error handling and Streamlit
UI integration for displaying errors.
"""

import requests
import streamlit as st

from config import REQUEST_TIMEOUT_SECONDS, build_api_url


JSON_HEADERS = {"accept": "application/json", "Content-Type": "application/json"}


def _request(method, path, **kwargs):
    """Make an HTTP request to the API and handle errors gracefully.

    Args:
        method (str): HTTP method ("GET", "POST", "DELETE", etc.).
        path (str): API endpoint path (e.g., "/chat").
        **kwargs: Additional arguments to pass to requests.request()
                 (e.g., json, files, headers, etc.).

    Returns:
        dict or None: Parsed JSON response if successful, None on error.
                     Errors are displayed via Streamlit UI.
    """
    try:
        response = requests.request(
            method,
            build_api_url(path),
            timeout=REQUEST_TIMEOUT_SECONDS,
            **kwargs,
        )
        response.raise_for_status()
        return response.json()
    except requests.HTTPError:
        st.error(f"API request failed with status code {response.status_code}: {response.text}")
        return None
    except requests.RequestException as e:
        st.error(f"Could not connect to the API server: {str(e)}")
        return None


def get_api_response(question, session_id, model):
    """Submit a chat question to the API and get an AI-generated response.

    Args:
        question (str): The user's question.
        session_id (str): Session identifier for conversation continuity.
        model (str): The model name to use (e.g., "gpt-4o-mini").

    Returns:
        dict or None: API response containing:
            - answer: The AI-generated response (str)
            - session_id: The session identifier (str)
            - model: The model used (str)
            Returns None if the request fails.
    """
    data = {"question": question, "model": model}
    if session_id:
        data["session_id"] = session_id

    return _request("POST", "/chat", headers=JSON_HEADERS, json=data)


def upload_document(file):
    """Upload a document file to the API for indexing.

    Args:
        file: Streamlit UploadedFile object with attributes:
            - name: Original filename (str)
            - file: File content (bytes)
            - type: MIME type (str)

    Returns:
        dict or None: API response containing:
            - message: Confirmation message (str)
            - file_id: Database ID of the indexed document (int)
            Returns None if the upload fails.
    """
    files = {"file": (file.name, file, file.type)}
    return _request("POST", "/upload-doc", files=files)


def list_documents():
    """Retrieve metadata for all uploaded documents from the API.

    Returns:
        list: List of document dictionaries with keys:
              - id: Document ID (int)
              - filename: Original filename (str)
              - upload_timestamp: Upload timestamp (datetime)
              Returns empty list if the request fails.
    """
    return _request("GET", "/documents") or []


def delete_document(file_id):
    """Delete a document from the API by ID.

    Args:
        file_id (int): The database ID of the document to delete.

    Returns:
        dict or None: API response containing:
            - message: Confirmation message (str)
            Returns None if the deletion fails.
    """
    return _request("DELETE", f"/documents/{file_id}", headers=JSON_HEADERS)
