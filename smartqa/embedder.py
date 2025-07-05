import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List


class Embedder:
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
    
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, show_progress_bar=True).astype('float32')
    
    def encode_single(self, text: str) -> np.ndarray:
        return self.model.encode([text]).astype('float32')
