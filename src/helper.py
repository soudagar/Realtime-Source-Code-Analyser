from dulwich import porcelain as git_porcelain
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
# Use HuggingFace Inference API embeddings — no torch/sentence-transformers needed (Vercel-safe)
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_chroma import Chroma
from src.config import settings
import os
import shutil


def clone_repo(repo_url: str):
    repo_path = settings.REPO_PATH
    # Remove stale clone if present so we always get a fresh copy
    if os.path.exists(repo_path):
        shutil.rmtree(repo_path)
    os.makedirs(repo_path, exist_ok=True)
    # dulwich is a pure-Python git impl — no system git binary needed (works on Vercel)
    git_porcelain.clone(repo_url, repo_path)
    return repo_path


def load_repo_docs(repo_path: str):
    documents = []
    language_map = {
        ".py":  Language.PYTHON,
        ".js":  Language.JS,
        ".ts":  Language.TS,
        ".jsx": Language.JS,
        ".tsx": Language.TS,
    }

    for ext, lang in language_map.items():
        try:
            loader = GenericLoader.from_filesystem(
                repo_path,
                glob=f"**/*{ext}",
                suffixes=[ext],
                parser=LanguageParser(language=lang, parser_threshold=500),
            )
            documents.extend(loader.load())
        except Exception as e:
            print(f"Error loading {ext} files: {e}")

    return documents


def split_docs(documents):
    splitters = {}
    text_chunks = []

    for doc in documents:
        lang_str = doc.metadata.get("language")
        if lang_str not in splitters:
            try:
                lang_enum = Language(lang_str)
                splitters[lang_str] = RecursiveCharacterTextSplitter.from_language(
                    language=lang_enum, chunk_size=2000, chunk_overlap=200
                )
            except (ValueError, TypeError):
                splitters[lang_str] = RecursiveCharacterTextSplitter(
                    chunk_size=2000, chunk_overlap=200
                )
        text_chunks.extend(splitters[lang_str].split_documents([doc]))

    return text_chunks


def load_embeddings():
    """Return HuggingFace Inference API embeddings — lightweight, no PyTorch required."""
    return HuggingFaceInferenceAPIEmbeddings(
        model_name=settings.EMBEDDING_MODEL,
    )


def build_vector_store(repo_url: str) -> Chroma:
    """Clone repo, parse, embed and return a fresh in-memory Chroma store."""
    repo_path = clone_repo(repo_url)
    documents = load_repo_docs(repo_path)
    text_chunks = split_docs(documents)

    if not text_chunks:
        raise ValueError(
            "No supported source files found in the repository "
            "(expected .py / .js / .ts / .jsx / .tsx)."
        )

    vector_store_path = settings.VECTOR_STORE_PATH
    # Wipe old store so embeddings are always fresh for the new repo
    if os.path.exists(vector_store_path):
        shutil.rmtree(vector_store_path)

    embeddings = load_embeddings()
    vector_store = Chroma.from_documents(
        text_chunks, embeddings, persist_directory=vector_store_path
    )
    print(f"Vector store created with {len(text_chunks)} chunks.")
    return vector_store
