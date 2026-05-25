# Model and Embeddings integrations
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
# Memory and Chains components remain in the langchain-classic package
from langchain_classic.memory import ConversationSummaryMemory
from langchain_classic.chains import ConversationalRetrievalChain
from src.helper import load_embeddings, build_vector_store
from src.config import settings
from pydantic import BaseModel, ValidationError
from flask import Flask, request, jsonify, render_template
import os
import shutil

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Global state — shared across warm lambda invocations on Vercel
# ---------------------------------------------------------------------------
vector_db = None
qa = None


def _init_qa(vdb: Chroma):
    """Build the conversational QA chain from a given vector store."""
    llm = ChatGroq(
        model=settings.MODEL_NAME,
        groq_api_key=settings.GROK_API_KEY,
        temperature=0.2,
    )
    memory = ConversationSummaryMemory(
        llm=llm, memory_key="chat_history", return_messages=True
    )
    return ConversationalRetrievalChain.from_llm(
        llm,
        retriever=vdb.as_retriever(search_type="mmr", search_kwargs={"k": 8}),
        memory=memory,
    )


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------
class RepoRequest(BaseModel):
    question: str  # re-uses existing field name (holds repo URL)


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    response: str
    success: bool = True


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")


@app.route("/chat", methods=["GET", "POST"])
def gitRepo():
    global vector_db, qa
    try:
        if request.method == "POST":
            data = RepoRequest(question=request.form["question"])
            repo_url = str(data.question)

            # Build vector store inline (no subprocess needed)
            vector_db = build_vector_store(repo_url)
            qa = _init_qa(vector_db)

            return jsonify({"response": f"Repo {repo_url} cloned and indexed successfully."})

        return jsonify({"error": "POST a repo URL via the 'question' field."}), 400

    except ValidationError as e:
        return jsonify({"error": str(e)}), 400
    except ValueError as e:
        return jsonify({"error": str(e)}), 422
    except Exception as e:
        error_msg = str(e)
        if "No valid credentials provided" in error_msg or "unauthorized" in error_msg.lower():
            return jsonify({
                "error": "Failed to index repository: The repository appears to be private or requires authentication. Only public repositories can be analyzed."
            }), 400
        return jsonify({"error": f"Failed to index repository: {error_msg}"}), 500


@app.route("/get", methods=["GET", "POST"])
def chat():
    global qa
    try:
        msg = request.form["question"]

        if msg.strip().lower() == "clear":
            # Clean up cloned repo; vector store stays until next /chat call
            repo_path = settings.REPO_PATH
            if os.path.exists(repo_path):
                shutil.rmtree(repo_path)
            return jsonify({"response": "Repository cleared.", "success": True})

        if qa is None:
            return jsonify({
                "response": "No repository indexed yet. Please submit a GitHub URL first.",
                "success": False,
            })

        result = qa({"question": msg})
        response = ChatResponse(response=result["answer"])
        print(result["answer"])
        return jsonify(response.model_dump())

    except Exception as e:
        return jsonify({"response": f"Error: {e}", "success": False}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
