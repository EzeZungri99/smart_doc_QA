import requests
import json
from typing import Any
from .chunker import Chunk


class LLMResponseGenerator:
    
    def __init__(self, model: str = "llama2", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
    
    def generate_response(self, query: str, relevant_chunks: list[tuple[Chunk, float]]) -> dict[str, Any]:
        if not relevant_chunks:
            return {
                "answer": "No relevant information found to answer your question.",
                "citations": [],
                "tokens_used": 0
            }
        
        context = self._build_context(relevant_chunks)
        prompt = f"Using ONLY the context below, answer the user's question. Keep your answer under 200 words.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 400
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get("response", "").strip()
                tokens_used = result.get("eval_count", 0)
            else:
                raise Exception(f"Ollama API error: {response.status_code}")
            
            citations = []
            for chunk, score in relevant_chunks:
                citations.append({
                    "chunk_id": chunk.id,
                    "text": chunk.text[:100] + "..." if len(chunk.text) > 100 else chunk.text,
                    "relevance_score": round(score, 3)
                })
            
            return {
                "answer": answer,
                "citations": citations,
                "tokens_used": tokens_used
            }
            
        except Exception as e:
            return {
                "answer": f"Error generating response: {str(e)}",
                "citations": [],
                "tokens_used": 0
            }
    
    def _build_context(self, relevant_chunks: list[tuple[Chunk, float]]) -> str:
        context_parts = []
        for i, (chunk, score) in enumerate(relevant_chunks, 1):
            context_parts.append(f"[{i}] {chunk.text}")
        return "\n\n".join(context_parts)
