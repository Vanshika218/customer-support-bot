import os
from langchain.text_splitter import RecursiveCharacterTextSplitter

data_folder = "customer_support_data"
all_chunks = []

for filename in os.listdir(data_folder):
    filepath = os.path.join(data_folder, filename)
    
    if filename.endswith(".txt"):
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read().strip()   # remove leading/trailing spaces
        
        print(f"\n--- {filename} content preview ---")
        print(text[:200])  # show first 200 characters
        print("-----------")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = text_splitter.split_text(text)
        print(f"{filename} → {len(chunks)} chunks")
        
        all_chunks.extend(chunks)

print(f"\nTotal chunks from all files: {len(all_chunks)}")
