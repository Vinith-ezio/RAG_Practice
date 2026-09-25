import json

from embeddings import generate_embeddings
from vector_store import load_index, search_index


class Retriever:

    def __init__(
        self,
        index_path="index/rag.index",
        chunks_path="index/chunks.json"
    ):

        # Load FAISS index
        self.index = load_index(index_path)

        # Load chunk metadata
        with open(
            chunks_path,
            "r",
            encoding="utf-8"
        ) as file:

            self.chunks = json.load(file)

    def search(self, query, top_k=3):

        # Convert query into embedding
        query_embedding = generate_embeddings(
            [query]
        )

        # Search vector index
        scores, indices = search_index(
            self.index,
            query_embedding,
            top_k=top_k
        )

        results = []

        for score, index_id in zip(
            scores,
            indices
        ):

            results.append({
                "chunk_id": int(index_id),
                "score": float(score),
                "text": self.chunks[index_id]["text"]
            })

        return results