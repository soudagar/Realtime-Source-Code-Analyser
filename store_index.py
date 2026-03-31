from src.helper import clone_repo, load_repo_docs, split_docs, load_embeddings
from langchain_chroma import Chroma
import os



# repo_url = "https://github.com/soudagar/AI-Book-Management-System.git"
# repo_path = clone_repo(repo_url)
documents = load_repo_docs("repo/")
text_chunks = split_docs(documents)
embeddings = load_embeddings()

if not text_chunks:
    print("No text chunks found. Please ensure the repository contains supported source code files (e.g., .py files).")
else:
    vector_store = Chroma.from_documents(text_chunks, embeddings, persist_directory="./vector_store")
    print("Vector store created successfully")
