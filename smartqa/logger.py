import json
import time
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path


class QALogger:
    
    def __init__(self, log_file: str = "qa_history.jsonl"):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(exist_ok=True)
    
    def log_interaction(
        self,
        query: str,
        answer: str,
        tokens_used: int,
        latency_ms: float,
        citations: list,
        input_file: str,
        model_name: str = "llama2",
        cost_usd: Optional[float] = None
    ) -> None:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "answer": answer,
            "tokens_used": tokens_used,
            "latency_ms": round(latency_ms, 2),
            "citations_count": len(citations),
            "input_file": input_file,
            "model_name": model_name,
            "cost_usd": cost_usd,
            "answer_length": len(answer),
            "query_length": len(query)
        }
        
        if citations:
            log_entry["citations"] = [
                {
                    "chunk_id": cit.get("chunk_id", "unknown"),
                    "relevance_score": cit.get("relevance_score", 0.0),
                    "text_preview": cit.get("text", "")[:100] + "..." if len(cit.get("text", "")) > 100 else cit.get("text", "")
                }
                for cit in citations
            ]
        
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    
    def get_stats(self) -> Dict[str, Any]:
        if not self.log_file.exists():
            return {
                "total_interactions": 0,
                "total_tokens": 0,
                "total_cost_usd": 0.0,
                "avg_latency_ms": 0.0
            }
        
        total_interactions = 0
        total_tokens = 0
        total_cost = 0.0
        total_latency = 0.0
        
        with open(self.log_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    total_interactions += 1
                    total_tokens += entry.get("tokens_used", 0)
                    
                    cost = entry.get("cost_usd")
                    if cost is not None:
                        total_cost += cost
                    
                    total_latency += entry.get("latency_ms", 0.0)
        
        return {
            "total_interactions": total_interactions,
            "total_tokens": total_tokens,
            "total_cost_usd": round(total_cost, 4),
            "avg_latency_ms": round(total_latency / max(total_interactions, 1), 2),
            "log_file": str(self.log_file)
        }
    
    def print_stats(self) -> None:
        stats = self.get_stats()
        
        print("\n📊 QA System Statistics:")
        print("=" * 40)
        print(f"Total interactions: {stats['total_interactions']}")
        print(f"Total tokens used: {stats['total_tokens']:,}")
        print(f"Total cost: ${stats['total_cost_usd']:.4f}")
        print(f"Average latency: {stats['avg_latency_ms']}ms")
        print(f"Log file: {stats['log_file']}")
        print("=" * 40)
