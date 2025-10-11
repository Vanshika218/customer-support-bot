import os
import pickle
from sentence_transformers import SentenceTransformer, util
import faiss
from langdetect import detect
from transformers import pipeline

# ----------------------------- Globals -----------------------------
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

# ----------------------------- Load FAISS & FAQ -----------------------------
def load_all():
    global index, chunks, embedder, rag_model, faq, faq_keys, faq_embeddings

    # Load FAISS index
    index = faiss.read_index("faiss_index.bin")

    # Load chunks
    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    # Load embedding model
    embedder = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")

    # Optional: Load RAG model via HuggingFace API
    HF_API_TOKEN = os.environ.get("HF_API_TOKEN")
    if HF_API_TOKEN:
        from huggingface_hub import InferenceClient
        rag_model = InferenceClient(HF_API_TOKEN)

    # Load FAQ
    def load_faq_from_txt(file_path):
        faq_dict = {}
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
            question, answer = None, None
            for line in lines:
                line = line.strip()
                if line.startswith("Q:"):
                    question = line[2:].strip()
                elif line.startswith("A:"):
                    answer = line[2:].strip()
                    if question and answer:
                        faq_dict[question.lower()] = answer
                        question, answer = None, None
        return faq_dict

    faq1_path = "customer_support_data/faq1.txt"
    faq2_path = "customer_support_data/faq2.txt"

    faq = {}
    if os.path.exists(faq1_path):
        faq.update(load_faq_from_txt(faq1_path))
    if os.path.exists(faq2_path):
        faq.update(load_faq_from_txt(faq2_path))

    faq_keys = list(faq.keys())
    if faq_keys:
        faq_embeddings = embedder.encode(faq_keys, convert_to_tensor=True)

# Load all data once
load_all()

# ----------------------------- Chatbot Function -----------------------------
def get_chatbot_response(query):
    global faq, faq_embeddings, faq_keys, index, chunks, embedder, rag_model

    try:
        detected_lang = detect(query)
    except:
        detected_lang = "en"

    translated_query = query
    if detected_lang != "en":
        translated_query = to_english(query)[0]['translation_text']

    # 1️⃣ FAQ check
    if faq_embeddings is not None and len(faq_embeddings) > 0:
        query_embedding = embedder.encode(translated_query, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_embedding, faq_embeddings)[0]
        best_idx = cos_scores.argmax().item()
        if cos_scores[best_idx] > 0.1:
            answer = faq[faq_keys[best_idx]]
            if detected_lang != "en":
                answer = from_english(answer)[0]['translation_text']
            return answer

    # 2️⃣ FAISS retrieval
    if index is not None:
        query_vec = embedder.encode([translated_query]).astype("float32")
        D, I = index.search(query_vec, 5)
        retrieved_texts = [chunks[idx] for idx in I[0] if idx != -1]
        context = " ".join(retrieved_texts[:2]) if retrieved_texts else ""

        if context and rag_model:
            prompt = f"""
You are a helpful support agent.
Answer clearly using ONLY the context.
If not in context, say "Sorry, I don’t have that information."

Context:
{context}

Question: {translated_query}
Answer:
"""
            response = rag_model.text2text(prompt)
            if detected_lang != "en":
                response = from_english(response)[0]['translation_text']
            return response

    # 3️⃣ Fallback
    fallback = "Sorry, I don’t have that information. Contact support@company.com."
    if detected_lang != "en":
        fallback = from_english(fallback)[0]['translation_text']
    return fallback
