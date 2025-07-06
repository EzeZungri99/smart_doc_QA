import requests
import json
import time
from typing import Any
from .chunker import Chunk
from .logger import QALogger


class LLMResponseGenerator:
    
    def __init__(self, model: str = "llama2", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self.logger = QALogger()
    
    def generate_response(self, query: str, relevant_chunks: list[tuple[Chunk, float]], input_file: str = "unknown") -> dict[str, Any]:
        start_time = time.time()
        
        if not relevant_chunks:
            return {
                "answer": "No relevant information found to answer your question.",
                "citations": [],
                "tokens_used": 0
            }
        
        context = self._build_context(relevant_chunks)
        prompt = (
            "Using ONLY the context below, answer the user's question. "
            "Keep your answer under 50 words. "
            "Answer in one or two sentences. "
            "Always respond in English.\n\n"
            f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        )
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 100
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
            
            latency_ms = (time.time() - start_time) * 1000
            
            self.logger.log_interaction(
                query=query,
                answer=answer,
                tokens_used=tokens_used,
                latency_ms=latency_ms,
                citations=citations,
                input_file=input_file,
                model_name=self.model
            )
            
            return {
                "answer": answer,
                "citations": citations,
                "tokens_used": tokens_used,
                "latency_ms": round(latency_ms, 2)
            }
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            
            self.logger.log_interaction(
                query=query,
                answer=f"Error: {str(e)}",
                tokens_used=0,
                latency_ms=latency_ms,
                citations=[],
                input_file=input_file,
                model_name=self.model
            )
            
            return {
                "answer": f"Error generating response: {str(e)}",
                "citations": [],
                "tokens_used": 0,
                "latency_ms": round(latency_ms, 2)
            }
    
    def _build_context(self, relevant_chunks: list[tuple[Chunk, float]]) -> str:
        context_parts = []
        for i, (chunk, score) in enumerate(relevant_chunks, 1):
            context_parts.append(f"[{i}] {chunk.text}")
        return "\n\n".join(context_parts)
    
    def print_stats(self) -> None:
        self.logger.print_stats()
