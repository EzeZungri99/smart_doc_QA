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
                "answer": "I don't have information about this topic in the provided document.",
                "citations": [],
                "tokens_used": 0,
                "latency_ms": 0
            }
        
        context = self._build_context(relevant_chunks)
        prompt = (
            "You are a document Q&A assistant. Your job is to answer questions based ONLY on the provided context.\n\n"
            "CRITICAL: If the question is about ANYTHING not explicitly mentioned in the context below, respond with:\n"
            "'I don't have information about this topic in the provided document.'\n\n"
            "This includes:\n"
            "- Personal questions (What's your favorite color?)\n"
            "- External facts (What's the weather in Uganda?)\n"
            "- Opinions (What do you think about...?)\n"
            "- Current events (What happened yesterday?)\n"
            "- Any topic not in the document\n\n"
            "Format: Under 50 words, 1-2 sentences, English only.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
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
                timeout=45
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get("response", "").strip()
                tokens_used = result.get("eval_count", 0)
                
                if not answer or answer.lower().startswith("i don't have") or "no information" in answer.lower():
                    answer = "I don't have information about this topic in the provided document."
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
