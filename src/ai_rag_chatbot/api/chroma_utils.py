"""Utils for loading, splitting, indexing, and deleting documents in Chroma."""

from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from typing import List
from langchain_core.documents import Document
import os

load_dotenv(Path(__file__).with_name(".env"))

# Initialize text splitter and embedding function
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
embedding_function = OpenAIEmbeddings()

# Initialize Chroma vector store
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)

def load_and_split_document(file_path: str) -> List[Document]:
    """Load a document from disk and split it into text chunks.

    The loader used depends on the file extension:
    - .pdf -> PyPDFLoader
    - .docx -> Docx2txtLoader
    - .html -> UnstructuredHTMLLoader

    The loaded document is then split using the shared
    RecursiveCharacterTextSplitter instance.

    Args:
        file_path: Path to the document file.

    Returns:
        A list of Document objects representing smaller chunks.

    Raises:
        ValueError: If the file extension is not supported.
    """
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith('.docx'):
        loader = Docx2txtLoader(file_path)
    elif file_path.endswith('.html'):
        loader = UnstructuredHTMLLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

    documents = loader.load()
    return text_splitter.split_documents(documents)


def index_document_to_chroma(file_path: str, file_id: int) -> bool:
    """Index a document's text chunks into the Chroma vector store.

    This function loads and splits the document first, then adds
    metadata for the originating file ID to every chunk before
    storing them in the shared vectorstore.

    Args:
        file_path: Path to the document file to index.
        file_id: Numeric identifier used to tag chunks from this file.

    Returns:
        True if indexing succeeds, False otherwise.
    """
    try:
        splits = load_and_split_document(file_path)

        # Add metadata to each split
        for split in splits:
            split.metadata['file_id'] = file_id

        vectorstore.add_documents(splits)
        return True
    except Exception as e:
        print(f"Error indexing document: {e}")
        return False


def delete_doc_from_chroma(file_id: int):
    """Delete all indexed chunks associated with a specific file ID.

    This function queries the Chroma collection for chunks tagged
    with the given file_id and removes them.

    Args:
        file_id: Numeric identifier of the file to remove.

    Returns:
        True if deletion succeeds, False otherwise.
    """
    try:
        docs = vectorstore.get(where={"file_id": file_id})
        print(f"Found {len(docs['ids'])} document chunks for file_id {file_id}")

        vectorstore._collection.delete(where={"file_id": file_id})
        print(f"Deleted all documents with file_id {file_id}")

        return True
    except Exception as e:
        print(f"Error deleting document with file_id {file_id} from Chroma: {str(e)}")
        return False
