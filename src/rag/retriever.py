import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class Retriever:
    def __init__(self, embeddings, chunks):
        self.embeddings = embeddings
        self.chunks = chunks

    def retrieve(self, query_embedding, top_k=2):
        similarities = cosine_similarity(
            [query_embedding],
            self.embeddings
        )[0]

        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [self.chunks[i] for i in top_indices]
