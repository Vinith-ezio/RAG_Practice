from pathlib import Path
import json

from chunking import chunk_text
from embeddings import generate_embeddings
from vector_store import create_index, save_index


# --------------------------------------------------
# 1. Load document
# --------------------------------------------------

file_path = Path("data/document.txt")

text = file_path.read_text(
    encoding="utf-8"
)


# --------------------------------------------------
# 2. Chunk document
# --------------------------------------------------

chunks = chunk_text(
    text,
    chunk_size=300,
    overlap=50
)

print("Total chunks:", len(chunks))


# --------------------------------------------------
# 3. Generate embeddings
# --------------------------------------------------

embeddings = generate_embeddings(chunks)

print("Embedding shape:", embeddings.shape)


# --------------------------------------------------
# 4. Create FAISS index
# --------------------------------------------------

index = create_index(embeddings)

print("Vectors stored:", index.ntotal)


# --------------------------------------------------
# 5. Save FAISS index
# --------------------------------------------------

save_index(
    index,
    "index/rag.index"
)

print("FAISS index saved!")


# --------------------------------------------------
# 6. Save chunk metadata
# --------------------------------------------------

chunk_data = []

for i, chunk in enumerate(chunks):

    chunk_data.append({
        "chunk_id": i,
        "text": chunk
    })


with open(
    "index/chunks.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        chunk_data,
        file,
        indent=4,
        ensure_ascii=False
    )


print("Chunk metadata saved!")