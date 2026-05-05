"""RAG chain implementation using LangChain for conversational AI with document retrieval.

This module implements a Retrieval-Augmented Generation (RAG) system that:
1. Retrieves relevant documents from a vector store
2. Reformulates user questions based on chat history
3. Generates AI responses augmented with retrieved context
"""

from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

from chroma_utils import vectorstore


def format_documents(documents: list[Document]) -> str:
    """Format a list of documents into a single readable text block.

    Args:
        documents (list[Document]): List of Document objects from the vector store.

    Returns:
        str: Formatted text with document contents joined by double newlines.
    """
    return "\n\n".join(doc.page_content for doc in documents)


def normalize_chat_history(chat_history: list[dict[str, str]]) -> list[Any]:
    """Convert chat history dictionaries into LangChain message objects.

    Args:
        chat_history (list[dict[str, str]]): List of message dicts with 'role' and 'content' keys.

    Returns:
        list[Any]: List of HumanMessage or AIMessage objects for use in prompts.
    """
    messages = []

    for message in chat_history:
        role = message.get("role")
        content = message.get("content", "")

        if role in {"human", "user"}:
            messages.append(HumanMessage(content=content))
        elif role in {"ai", "assistant"}:
            messages.append(AIMessage(content=content))

    return messages


contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Given a chat history and the latest user question, "
     "formulate a standalone question which can be understood "
     "without the chat history. Do NOT answer the question."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

qa_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful AI assistant. Use the following context to answer the user's question.\n\n"
     "Context:\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])


class RAGChain:
    """A Retrieval-Augmented Generation chain for conversational AI.

    This class orchestrates the RAG process: reformulating questions based on
    chat history, retrieving relevant documents, and generating AI responses
    augmented with retrieved context.
    """

    def __init__(self, model: str = "gpt-4o-mini", k: int = 2):
        """Initialize the RAG chain with an LLM and document retriever.

        Args:
            model (str): The OpenAI model to use (default: "gpt-4o-mini").
            k (int): Number of documents to retrieve (default: 2).
        """
        self.llm = ChatOpenAI(model=model)
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        self.contextualize_question_chain = contextualize_q_prompt | self.llm | StrOutputParser()
        self.answer_chain = qa_prompt | self.llm | StrOutputParser()

    def invoke(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Execute the RAG pipeline on user input.

        Process:
        1. Extract the current question and normalize chat history
        2. If history exists, reformulate the question to be standalone
        3. Retrieve relevant documents using the (possibly reformulated) question
        4. Generate an answer using the documents as context

        Args:
            inputs (dict[str, Any]): Dict with keys:
                - "input": The user's question (str)
                - "chat_history": Optional prior messages (list[dict])

        Returns:
            dict[str, Any]: Dict with keys:
                - "answer": The AI-generated response (str)
                - "context": Retrieved documents (list[Document])
        """
        question = inputs["input"]
        chat_history = normalize_chat_history(inputs.get("chat_history", []))

        if chat_history:
            retrieval_query = self.contextualize_question_chain.invoke({
                "input": question,
                "chat_history": chat_history,
            })
        else:
            retrieval_query = question

        documents = self.retriever.invoke(retrieval_query)

        answer = self.answer_chain.invoke({
            "input": question,
            "chat_history": chat_history,
            "context": format_documents(documents),
        })

        return {
            "answer": answer,
            "context": documents,
        }


def get_rag_chain(model: str = "gpt-4o-mini") -> RAGChain:
    """Factory function to create and return a RAGChain instance.

    Args:
        model (str): The OpenAI model to use (default: "gpt-4o-mini").

    Returns:
        RAGChain: A configured RAG chain instance ready to invoke.
    """
    return RAGChain(model=model)
