import os
import pickle
import faiss
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

# -----------------------------
# Flask App Setup
# -----------------------------
app = Flask(__name__)
app.secret_key = "supersecretkey"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///chat.db"
db = SQLAlchemy(app)

# -----------------------------
# Database Models
# -----------------------------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

class ChatHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    user_message = db.Column(db.Text, nullable=False)
    bot_message = db.Column(db.Text, nullable=False)

# -----------------------------
# Globals for models/data
# -----------------------------
index = None
chunks = None
embedder = None
rag_model = None
faq = {}
faq_keys = []
faq_embeddings = None

# Translation pipelines
to_english = pipeline("translation", model="Helsinki-NLP/opus-mt-mul-en")
from_english = pipeline("translation", model="Helsinki-NLP/opus-mt-en-mul")

# -----------------------------
# Load FAQ, Models, FAISS
# -----------------------------
def load_all():
    global index, chunks, embedder, rag_model, faq, faq_keys, faq_embeddings

    index = faiss.read_index("faiss_index.bin")
    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    rag_model = pipeline("text2text-generation", model="google/flan-t5-base")

    def load_faq_from_txt(file_path):
        faq_dict = {}
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
            question, answer = None, None
            for line in lines:
                if line.startswith("Q:"):
                    question = line[2:].strip()
                elif line.startswith("A:"):
                    answer = line[2:].strip()
                    if question and answer:
                        faq_dict[question.lower()] = answer
                        question, answer = None, None
        return faq_dict

    faq1_path = r"C:\Users\VANSHIKA\OneDrive\Desktop\customer_support_bot\customer_support_data\faq1.txt"
    faq2_path = r"C:\Users\VANSHIKA\OneDrive\Desktop\customer_support_bot\customer_support_data\faq2.txt"

    faq = {}
    if os.path.exists(faq1_path):
        faq.update(load_faq_from_txt(faq1_path))
    if os.path.exists(faq2_path):
        faq.update(load_faq_from_txt(faq2_path))

    faq_keys = list(faq.keys())
    faq_embeddings = embedder.encode(faq_keys, convert_to_tensor=True)

    return True

load_all()

# -----------------------------
# Chatbot Response
# -----------------------------
def get_chatbot_response(query, user_id):
    from langdetect import detect

    original_query = query
    try:
        detected_lang = detect(query)
    except:
        detected_lang = "en"

    translated_query = query
    if detected_lang != "en":
        translated_query = to_english(query)[0]['translation_text']

    # FAQ match
    if faq_embeddings is not None and len(faq_embeddings) > 0:
        query_embedding = embedder.encode(translated_query, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_embedding, faq_embeddings)[0]
        best_idx = cos_scores.argmax().item()
        if cos_scores[best_idx] > 0.1:
            answer = faq[faq_keys[best_idx]]
            final_answer = answer
            if detected_lang != "en":
                final_answer = from_english(answer)[0]['translation_text']
            db.session.add(ChatHistory(user_id=user_id, user_message=original_query, bot_message=final_answer))
            db.session.commit()
            return final_answer

    # FAISS retrieval
    query_vec = embedder.encode([translated_query])
    D, I = index.search(query_vec, 5)
    if len(I[0]) > 0 and I[0][0] != -1:
        retrieved_texts = [chunks[idx] for idx in I[0] if idx != -1]
        context = " ".join(retrieved_texts[:2])

        prompt = f"""
You are a helpful customer support agent.
Answer clearly using ONLY the context.
If not in context, say "Sorry, I don’t have that information."

Context:
{context}

Question: {translated_query}
Answer:
"""
        response = rag_model(prompt, max_new_tokens=150, do_sample=False)[0]['generated_text'].strip()
        final_answer = response
        if detected_lang != "en":
            final_answer = from_english(response)[0]['translation_text']
        db.session.add(ChatHistory(user_id=user_id, user_message=original_query, bot_message=final_answer))
        db.session.commit()
        return final_answer

    fallback = (
        "Sorry, I don’t have that information. "
        "You can contact support at support@company.com or call 1-800-123-4567."
    )
    final_answer = fallback
    if detected_lang != "en":
        final_answer = from_english(fallback)[0]['translation_text']
    db.session.add(ChatHistory(user_id=user_id, user_message=original_query, bot_message=final_answer))
    db.session.commit()
    return final_answer

# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def index():
    return render_template("index.html", username=session.get("username"))

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form["username"]
        password = generate_password_hash(request.form["password"])
        if User.query.filter_by(username=username).first():
            return "User already exists!"
        user = User(username=username, password=password)
        db.session.add(user)
        db.session.commit()
        return redirect(url_for("login"))
    return render_template("signup.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            session["user_id"] = user.id
            session["username"] = user.username
            return redirect(url_for("index"))
        return "Invalid credentials!"
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))

@app.route("/chat", methods=["POST"])
def chat():
    if "user_id" not in session:
        return jsonify({"response": "Please log in first!"})
    data = request.json
    user_message = data.get("message")
    bot_response = get_chatbot_response(user_message, session["user_id"])
    return jsonify({"response": bot_response})

@app.route("/history")
def history():
    if "user_id" not in session:
        return redirect(url_for("login"))
    history = ChatHistory.query.filter_by(user_id=session["user_id"]).all()
    return render_template("history.html", history=history)

# -----------------------------
# Run
# -----------------------------

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
