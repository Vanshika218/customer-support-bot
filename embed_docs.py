import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np

DATA_FOLDER = "customer_support_data"
FAISS_INDEX_FILE = "faiss_index.bin"
CHUNKS_FILE = "chunks.pkl"
BATCH_SIZE = 50  # Number of chunks per batch
EMBEDDING_MODEL = "sentence-transformers/paraphrase-MiniLM-L6-v2"  # small & CPU-friendly
EMBEDDING_DIM = 384  # dimension for MiniLM-L6-v2

print("Loading embedding model...")
model = SentenceTransformer(EMBEDDING_MODEL)

all_chunks = []

# Step 1: Read & chunk files
for filename in os.listdir(DATA_FOLDER):
    if filename.endswith(".txt"):
        filepath = os.path.join(DATA_FOLDER, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read().strip()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = text_splitter.split_text(text)
        all_chunks.extend(chunks)  # ✅ store only text

print(f"Total text chunks: {len(all_chunks)}")

# Step 2: Create FAISS index
index = faiss.IndexFlatL2(EMBEDDING_DIM)

# Step 3: Encode in batches
print("Encoding chunks in batches...")
for i in range(0, len(all_chunks), BATCH_SIZE):
    batch_chunks = all_chunks[i:i + BATCH_SIZE]
    embeddings = model.encode(batch_chunks, convert_to_numpy=True, show_progress_bar=False)
    embeddings = embeddings.astype("float32")  # reduce memory
    index.add(embeddings)
    print(f"Processed batch {i // BATCH_SIZE + 1} / {(len(all_chunks) - 1) // BATCH_SIZE + 1}")

# Step 4: Save index & chunks
faiss.write_index(index, FAISS_INDEX_FILE)
with open(CHUNKS_FILE, "wb") as f:
    pickle.dump(all_chunks, f)

print("✅ FAISS index and chunks saved successfully!")
