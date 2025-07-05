import faiss
from typing import List, Tuple
from .chunker import Chunk
from .embedder import Embedder


class Retriever:
    
    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.index = None
        self.chunks: List[Chunk] = []
    
    def add_chunks(self, chunks: List[Chunk]):
        if not chunks:
            return
        
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedder.encode_texts(texts)
        
        if self.index is None:
            self.index = faiss.IndexFlatIP(self.embedder.dimension)
        
        self.index.add(embeddings)
        self.chunks.extend(chunks)
    
    def search(self, query: str, k: int = 5) -> List[Tuple[Chunk, float]]:
        if self.index is None or len(self.chunks) == 0:
            return []
        
        query_embedding = self.embedder.encode_single(query)
        scores, indices = self.index.search(query_embedding, min(k, len(self.chunks)))
        
        results = []
        for i, score in zip(indices[0], scores[0]):
            if i < len(self.chunks):
                results.append((self.chunks[i], float(score)))
        
        return results
