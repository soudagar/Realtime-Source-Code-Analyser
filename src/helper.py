from git import Repo
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
# Model and Embeddings integrations are now in langchain_openai
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
# Chroma has its own dedicated community package now
from langchain_chroma import Chroma
# Memory and Chains components remain in the core langchain package
from langchain_classic.memory import ConversationSummaryMemory
from langchain_classic.chains import ConversationalRetrievalChain
import os

def clone_repo(repo_url: str):
    repo_path = "repo"
    if not os.path.exists(repo_path):
        Repo.clone_from(repo_url, repo_path)
    return repo_path

def load_repo_docs(repo_path: str):
    documents = []
    language_map = {
        ".py": Language.PYTHON,
        ".js": Language.JS,
        ".ts": Language.TS,
        ".jsx": Language.JS,
        ".tsx": Language.TS
    }
    
    for ext, lang in language_map.items():
        try:
            loader = GenericLoader.from_filesystem(
                repo_path,
                glob=f"**/*{ext}", 
                suffixes=[ext],
                parser=LanguageParser(language=lang, parser_threshold=500)
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
                splitters[lang_str] = RecursiveCharacterTextSplitter.from_language(language=lang_enum, chunk_size=2000, chunk_overlap=200)
            except (ValueError, TypeError):
                splitters[lang_str] = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
                
        text_chunks.extend(splitters[lang_str].split_documents([doc]))
        
    return text_chunks

def load_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings
