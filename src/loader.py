from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_core.documents import Document
from typing import List
from config import BOOKS_DIR

def load_document(books_dir: str = BOOKS_DIR) -> List[Document]:
    print(f"loading documents from {books_dir}")
    all_docs = []

    loader = DirectoryLoader(
        path = books_dir,
        glob = "**/*.pdf",
        loader_cls = PyMuPDFLoader,
        use_multithreading = True,
        show_progress = True
)
    for doc in loader.lazy_load():
        all_docs.append(doc)

    print(f'loaded {len(all_docs)} pages successfully')
    return all_docs