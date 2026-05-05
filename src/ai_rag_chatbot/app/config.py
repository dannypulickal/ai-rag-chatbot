"""Configuration for the Streamlit chat application.

Manages API connection settings and URL construction.
"""

import os
from urllib.parse import urljoin

import streamlit as st


DEFAULT_API_BASE_URL = "http://localhost:8000"
API_BASE_URL_ENV_VAR = "RAG_API_BASE_URL"
REQUEST_TIMEOUT_SECONDS = 30


def get_api_base_url() -> str:
    """Retrieve the API base URL from environment, secrets, or default.

    Priority (highest to lowest):
    1. Environment variable RAG_API_BASE_URL
    2. Streamlit secrets (api_base_url)
    3. Default value (http://localhost:8000)

    Returns:
        str: The API base URL without trailing slash.
    """
    api_base_url = os.getenv(API_BASE_URL_ENV_VAR)

    if not api_base_url:
        try:
            api_base_url = st.secrets.get("api_base_url")
        except Exception:
            api_base_url = None

    if not api_base_url:
        api_base_url = DEFAULT_API_BASE_URL

    return api_base_url.rstrip("/")


def build_api_url(path: str) -> str:
    """Build a full API URL from a path.

    Args:
        path (str): The API endpoint path (e.g., "/chat").

    Returns:
        str: The full API URL (e.g., "http://localhost:8000/chat").
    """
    return urljoin(f"{get_api_base_url()}/", path.lstrip("/"))
