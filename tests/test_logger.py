import pytest
import json
import os
from smartqa.logger import QALogger


def test_logger_creation():
    logger = QALogger()
    assert hasattr(logger, 'log_file')
    assert str(logger.log_file).endswith('.jsonl')


def test_log_interaction():
    logger = QALogger()
    
    logger.log_interaction(
        query="What is AI?",
        answer="AI is artificial intelligence.",
        tokens_used=45,
        latency_ms=1234.56,
        citations=[],
        input_file="test.txt",
        model_name="llama2"
    )
    
    assert os.path.exists(logger.log_file)
    
    with open(logger.log_file, 'r') as f:
        lines = f.readlines()
        assert len(lines) > 0
        
        last_entry = json.loads(lines[-1])
        assert last_entry["query"] == "What is AI?"
        assert last_entry["answer"] == "AI is artificial intelligence."
        assert last_entry["tokens_used"] == 45
        assert last_entry["latency_ms"] == 1234.56


def test_logger_multiple_interactions():
    logger = QALogger()
    
    interactions = [
        {"query": "What is AI?", "answer": "AI is artificial intelligence.", "tokens": 10, "latency": 500},
        {"query": "What is ML?", "answer": "ML is machine learning.", "tokens": 8, "latency": 400}
    ]
    
    for interaction in interactions:
        logger.log_interaction(
            query=interaction["query"],
            answer=interaction["answer"],
            tokens_used=interaction["tokens"],
            latency_ms=interaction["latency"],
            citations=[],
            input_file="test.txt",
            model_name="llama2"
        )
    
    with open(logger.log_file, 'r') as f:
        lines = f.readlines()
        assert len(lines) >= len(interactions)


def test_logger_stats():
    logger = QALogger()
    
    test_interactions = [
        {"tokens_used": 10, "latency_ms": 500},
        {"tokens_used": 15, "latency_ms": 600},
        {"tokens_used": 20, "latency_ms": 700}
    ]
    
    for interaction in test_interactions:
        logger.log_interaction(
            query="test",
            answer="test answer",
            tokens_used=interaction["tokens_used"],
            latency_ms=interaction["latency_ms"],
            citations=[],
            input_file="test.txt",
            model_name="llama2"
        )
    
    stats = logger.get_stats()
    
    assert "total_interactions" in stats
    assert "total_tokens" in stats
    assert "avg_latency_ms" in stats
    assert stats["total_interactions"] >= len(test_interactions)
    assert stats["total_tokens"] >= sum(i["tokens_used"] for i in test_interactions)


def test_logger_empty_stats():
    logger = QALogger()
    
    if os.path.exists(logger.log_file):
        os.remove(logger.log_file)
    
    stats = logger.get_stats()
    
    assert stats["total_interactions"] == 0
    assert stats["total_tokens"] == 0
    assert stats["avg_latency_ms"] == 0 