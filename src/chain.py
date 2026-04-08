from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_classic.retrievers import ContextualCompressionRetriever
from config import (
    GROQ_MODEL,
    TEMPERATURE,
    MAX_TOKENS,
    TOP_P,
    MEMORY_K)
import os

def get_llm() -> ChatGroq:
    """
    Returns configures groq LLM instances

    """
    llm = ChatGroq(
        model = GROQ_MODEL,
        temperature = TEMPERATURE,
        max_tokens = MAX_TOKENS,
        streaming = True,
        api_key = os.getenv("GROQ_API_KEY")
    )

    return llm

def get_memory()-> ConversationBufferWindowMemory:
    """
    Returns conversation memory
    remembers last k exchanges only

    """
    memory = ConversationBufferWindowMemory(
        k = MEMORY_K,
        memory_key = "chat_history",
        return_messages = True,
        output_key = "answer",
        human_prefix = "Student",
        ai_prefix = "Tutor"
    )
    return memory

def get_chain(
        retriever: ContextualCompressionRetriever,
        prompt: PromptTemplate
) -> ConversationalRetrievalChain:
    """
    Builds and returns the full conversational RAG chain
    combines LLM + retriever + memory + prompt

    """
    llm = get_llm()
    memory = get_memory()

    chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = retriever,
        memory = memory,
        return_source_documents = True,
        combine_docs_chain_kwargs = {"prompt": prompt},
        verbose = False
    )

    print("chain ready")
    return chain