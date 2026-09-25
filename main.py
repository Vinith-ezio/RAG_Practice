from pathlib import Path

from chunking import chunk_text
from embeddings import generate_embeddings
from embeddings import model

from vector_store import create_index, search_index


# --------------------------------------------------
# 1. Load document
# --------------------------------------------------

file_path = Path("data/document.txt")

text = file_path.read_text(
    encoding="utf-8"
)

print("Document characters:", len(text))


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
# 3. Generate document embeddings
# --------------------------------------------------

embeddings = generate_embeddings(chunks)

print("Embedding shape:", embeddings.shape)


# --------------------------------------------------
# 4. Create FAISS index
# --------------------------------------------------

index = create_index(embeddings)

print("FAISS index created!")
print("Vectors stored:", index.ntotal)
print("Vector dimension:", index.d)


# --------------------------------------------------
# 5. User query
# --------------------------------------------------

query = "How the energy consumption is measured by?"

print("\nQuery:")
print(query)


# --------------------------------------------------
# 6. Generate query embedding
# --------------------------------------------------

query_embedding = generate_embeddings([query])


# --------------------------------------------------
# 7. Search FAISS
# --------------------------------------------------

scores, indices = search_index(
    index,
    query_embedding,
    top_k=3
)


# --------------------------------------------------
# 8. Display results
# --------------------------------------------------

print("\n" + "=" * 70)
print("FAISS SEARCH RESULTS")
print("=" * 70)

for rank, (score, index_id) in enumerate(
    zip(scores, indices),
    start=1
):

    print(f"\nRank: {rank}")
    print(f"Chunk ID: {index_id}")
    print(f"Similarity Score: {score:.4f}")

    print("\nChunk:")
    print(chunks[index_id])

    print("-" * 70)