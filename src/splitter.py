from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List
from config import CHUNK_SIZE,CHUNK_OVERLAP


def get_splitter() -> RecursiveCharacterTextSplitter:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = CHUNK_SIZE,
        chunk_overlap = CHUNK_OVERLAP,
        separators = ["\n\n",      # paragraph breaks
                "\n",        # line breaks
                "```",       # code blocks
                ". ",        # sentences
                 " ",         # words
                  ""  ]         # characters (last resort)
    )
    return splitter

def split_document(docs: List[Document]) -> List[Document]:
    """
        Splits cleaned documents into chunks
        Preserves metadata from original documents
        """
    splitter = get_splitter()
    chunks = splitter.split_documents(docs)

    print(f"Created {len(chunks)} chunks successfully.")
    return chunks


