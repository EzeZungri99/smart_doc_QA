import os
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List


class Embedder:
    def __init__(self, model_name=None):
        self.model_name = model_name or os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
        self.model = SentenceTransformer(self.model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def encode_single(self, text):
        return self.model.encode([text], convert_to_numpy=True)

    def encode_texts(self, texts):
        return self.model.encode(texts, convert_to_numpy=True)
