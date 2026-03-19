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
    loader = GenericLoader.from_filesystem(repo_path,
                                       glob="**/*",
                                       suffixes=[".py"],
                                       parser = LanguageParser(language="python", parser_threshold=500)
                                       )
    documents = loader.load()
    return documents

def split_docs(documents):
    text_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.PYTHON, chunk_size=2000, chunk_overlap=200)
    text_chunk = text_splitter.split_documents(documents)  
    return text_chunk

def load_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings
