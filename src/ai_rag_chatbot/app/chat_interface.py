"""Streamlit chat interface component.

Provides the main chat UI where users can:
- View chat history
- Send queries to the RAG chatbot
- See AI responses and generation details
"""

import streamlit as st
from api_utils import get_api_response


def display_chat_interface():
    """Display an interactive chat interface in Streamlit.

    Manages:
    1. Rendering prior chat messages
    2. Accepting user input via chat_input
    3. Calling the API to get AI responses
    4. Updating session state with new messages
    5. Displaying responses with collapsible details

    Expects the following in st.session_state:
    - messages: List of message dicts with 'role' and 'content'
    - session_id: Unique session identifier for conversation continuity
    - model: The AI model to use for responses

    Updates st.session_state.session_id with the API response to maintain
    conversation context across Streamlit reruns.
    """
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle new user input
    if prompt := st.chat_input("Query:"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get API response
        with st.spinner("Generating response..."):
            response = get_api_response(prompt, st.session_state.session_id, st.session_state.model)

            if response:
                st.session_state.session_id = response.get('session_id')
                st.session_state.messages.append({"role": "assistant", "content": response['answer']})

                with st.chat_message("assistant"):
                    st.markdown(response['answer'])

                with st.expander("Details"):
                    st.subheader("Generated Answer")
                    st.code(response['answer'])
                    st.subheader("Model Used")
                    st.code(response['model'])
                    st.subheader("Session ID")
                    st.code(response['session_id'])
            else:
                st.error("Failed to get a response from the API. Please try again.")
