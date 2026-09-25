import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def retrieve(query, chunks, chunk_embeddings, embedding_model, top_k=3):

    # Convert user query into an embedding
    query_embedding = embedding_model.encode([query])

    # Calculate similarity between query and every chunk
    scores = cosine_similarity(
        query_embedding,
        chunk_embeddings
    )[0]

    # Get indices of highest similarity scores
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for index in top_indices:
        results.append({
            "chunk_id": index,
            "score": scores[index],
            "text": chunks[index]
        })

    return results