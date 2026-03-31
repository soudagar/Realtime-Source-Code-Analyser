# Model and Embeddings integrations
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
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

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")


@app.route("/chat", methods=["GET", "POST"])
def gitRepo():
    if request.method == "POST":
        user_msg = request.form["question"]
        clone_repo(user_msg)
        os.system("python store_index.py")
    return jsonify({"response": f"Repo {user_msg} cloned successfully"})

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["question"]
    input = msg
    print(input)
    
    if input == 'clear':
        os.system("rm -rf repo")
        ## os.system("rm -rf vector_store")

    result = qa({"question": input})
    print(result["answer"])
    return jsonify({"response": result["answer"]}) 




if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
