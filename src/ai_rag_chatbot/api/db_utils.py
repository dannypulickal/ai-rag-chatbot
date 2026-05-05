"""SQLite helpers for chat logs and uploaded document metadata."""

import sqlite3

DB_NAME = "rag_app.db"


def get_db_connection():
    """Create a SQLite connection configured to return rows as dictionaries.

    Returns:
        sqlite3.Connection: A connection object with row_factory set to sqlite3.Row.
    """
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn


def create_application_logs():
    """Create the application log table if it does not already exist.

    This table stores chat exchanges for each session, including the user
    query, the AI response, the model used, and a timestamp.
    """
    conn = get_db_connection()
    conn.execute('''CREATE TABLE IF NOT EXISTS application_logs
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     session_id TEXT,
                     user_query TEXT,
                     gpt_response TEXT,
                     model TEXT,
                     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    conn.close()


def create_document_store():
    """Create the uploaded document metadata table if it does not already exist.

    This table stores information about files uploaded for indexing, including
    the original filename and upload time.
    """
    conn = get_db_connection()
    conn.execute('''CREATE TABLE IF NOT EXISTS document_store
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     filename TEXT,
                     upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    conn.close()


def insert_application_logs(session_id, user_query, gpt_response, model):
    """Persist one chat exchange for a session.

    Args:
        session_id (str): A unique session identifier for the chat.
        user_query (str): The user's input question.
        gpt_response (str): The AI model's response text.
        model (str): The name of the model used to generate the response.
    """
    conn = get_db_connection()
    conn.execute('INSERT INTO application_logs (session_id, user_query, gpt_response, model) VALUES (?, ?, ?, ?)',
                 (session_id, user_query, gpt_response, model))
    conn.commit()
    conn.close()


def get_chat_history(session_id):
    """Return prior chat messages for a session in LangChain message format.

    Args:
        session_id (str): The session identifier to fetch history for.

    Returns:
        list[dict[str, str]]: A list of message dictionaries formatted for LangChain.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT user_query, gpt_response FROM application_logs WHERE session_id = ? ORDER BY created_at', (session_id,))
    messages = []
    for row in cursor.fetchall():
        messages.extend([
            {"role": "human", "content": row['user_query']},
            {"role": "ai", "content": row['gpt_response']}
        ])
    conn.close()
    return messages


def insert_document_record(filename):
    """Insert uploaded document metadata and return the generated document ID.

    Args:
        filename (str): The original name of the uploaded file.

    Returns:
        int: The auto-generated database ID for the new document.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('INSERT INTO document_store (filename) VALUES (?)', (filename,))
    file_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return file_id


def delete_document_record(file_id):
    """Delete uploaded document metadata by document ID.

    Args:
        file_id (int): The database ID of the document to delete.

    Returns:
        bool: True when the deletion is executed.
    """
    conn = get_db_connection()
    conn.execute('DELETE FROM document_store WHERE id = ?', (file_id,))
    conn.commit()
    conn.close()
    return True


def get_all_documents():
    """Return all uploaded document metadata ordered from newest to oldest.

    Returns:
        list[dict[str, Any]]: A list of document metadata dictionaries.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT id, filename, upload_timestamp FROM document_store ORDER BY upload_timestamp DESC')
    documents = cursor.fetchall()
    conn.close()
    return [dict(doc) for doc in documents]

# Initialize the database tables
create_application_logs()
create_document_store()
