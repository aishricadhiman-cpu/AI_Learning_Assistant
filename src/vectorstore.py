from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from typing import List
from config import CHROMA_DIR

def create_vectorstore(
        chunks: List[Document],
        embeddings: HuggingFaceEmbeddings,
        persist_dir: str = CHROMA_DIR
) -> Chroma:
    """
      Creates ChromaDB vectorstore from chunks
      Embeds all chunks and persists to disk
      Run once during ingestion only
      """

    print("creating chromadb vectorstore..")

    vectorstore = Chroma.from_documents(
        documents = chunks,
        embedding = embeddings,
        collection_name = "ml_dl_tutor",
        persist_directory = persist_dir
    )
    print("created chromadb vectorstore")
    return vectorstore

def load_vectorstore(
        embeddings: HuggingFaceEmbeddings,
        persist_dir: str = CHROMA_DIR
) -> Chroma:
    """
        Loads existing ChromaDB from disk
        No re-embedding — instant load
        Used in main.py every session
    """
    vectorstore = Chroma(
        collection_name = "ml_dl_tutor",
        embedding_function = embeddings,
        persist_directory = persist_dir
    )

    print("Vectorstore loaded succesfully")
    return vectorstore







