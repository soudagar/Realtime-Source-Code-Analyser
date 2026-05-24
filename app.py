# Model and Embeddings integrations
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from pydantic import BaseModel, HttpUrl, ValidationError
# Chroma has its own dedicated community package now
from langchain_chroma import Chroma
# Memory and Chains components remain in the core langchain package
from langchain_classic.memory import ConversationSummaryMemory
from langchain_classic.chains import ConversationalRetrievalChain
from src.helper import clone_repo, load_embeddings
from flask import Flask, request, jsonify, render_template
import os

app = Flask(__name__)
    
embedding = load_embeddings()
vector_db = Chroma(persist_directory='./vector_store', embedding_function=embedding)

llm = ChatOllama(model="llama3.2:latest")
memory = ConversationSummaryMemory(llm=llm, memory_key="chat_history", return_messages=True)

qa = ConversationalRetrievalChain.from_llm(llm, retriever=vector_db.as_retriever(search_type="mmr", search_kwargs={"k":8}),memory=memory)

class RepoRequest(BaseModel):
    question: str
    
class ChatRequest(BaseModel):
    question:str
    
class ChatResponse(BaseModel):
    response: str
    success: bool= True
    


@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")


@app.route("/chat", methods=["GET", "POST"])
def gitRepo():
    try:
        if request.method == "POST":
            data = RepoRequest(question=request.form["question"])
            clone_repo(str(data.question))
            os.system("python store_index.py")
            return jsonify({"response": f"Repo {data.question} cloned successfully"})
    except ValidationError as e:
        return jsonify({"error": str(e)}), 400

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["question"]
    input = msg
    print(input)
    
    if input == 'clear':
        os.system("rm -rf repo")
        ## os.system("rm -rf vector_store")

    result = qa({"question": input})
    response = ChatResponse(response = result["answer"])
    print(result["answer"])
    return jsonify(response.model_dump()) 




if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
