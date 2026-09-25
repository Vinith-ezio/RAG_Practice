import faiss
import numpy as np


def create_index(embeddings):

    embeddings = embeddings.astype("float32")

    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index


def search_index(index, query_embedding, top_k=3):

    query_embedding = query_embedding.astype("float32")

    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(
        query_embedding,
        top_k
    )

    return scores[0], indices[0]


def save_index(index, path):

    faiss.write_index(
        index,
        path
    )


def load_index(path):

    return faiss.read_index(path)