import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

data_folder = "customer_support_data"
all_chunks = []

# Step 1: Chunk all text files
for filename in os.listdir(data_folder):
    if filename.endswith(".txt"):
        filepath = os.path.join(data_folder, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read().strip()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = text_splitter.split_text(text)
        all_chunks.extend(chunks)  # ✅ only add text

print(f"Total chunks: {len(all_chunks)}")

# Step 2: Encode chunks
embeddings = model.encode(all_chunks, convert_to_numpy=True, show_progress_bar=True)

# Step 3: Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

print(f"FAISS index created with {index.ntotal} vectors")

# Step 4: Save FAISS index and chunks
faiss.write_index(index, "faiss_index.bin")
with open("chunks.pkl", "wb") as f:
    pickle.dump(all_chunks, f)  # ✅ only text

print("✅ Saved FAISS index and chunks")
