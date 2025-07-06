import json
import tempfile
import os
from pathlib import Path
from smartqa.logger import QALogger


def test_logger_creation():
    """Test logger creation and initialization"""
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp_file:
        log_file = tmp_file.name
    
    try:
        logger = QALogger(log_file)
        assert logger.log_file == Path(log_file)
        print("✅ Logger creation test PASSED")
    finally:
        if os.path.exists(log_file):
            os.unlink(log_file)


def test_log_interaction():
    """Test logging a single interaction"""
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp_file:
        log_file = tmp_file.name
    
    try:
        logger = QALogger(log_file)
        
        # Log a test interaction
        test_citations = [
            {
                "chunk_id": "chunk_1",
                "text": "Test citation text",
                "relevance_score": 0.85
            }
        ]
        
        print(f"   📝 Logging interaction...")
        print(f"   Query: 'What is AI?'")
        print(f"   Answer: 'AI is artificial intelligence'")
        print(f"   Tokens: 45")
        print(f"   Latency: 123.45ms")
        print(f"   Citations: 1")
        print(f"   Input file: test.txt")
        print(f"   Model: llama2")
        
        logger.log_interaction(
            query="What is AI?",
            answer="AI is artificial intelligence",
            tokens_used=45,
            latency_ms=123.45,
            citations=test_citations,
            input_file="test.txt",
            model_name="llama2"
        )
        
        # Verify the log file was created and contains the entry
        assert os.path.exists(log_file)
        
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            assert len(lines) == 1
            
            entry = json.loads(lines[0])
            print(f"\n   📊 Logged data:")
            print(f"   Timestamp: {entry['timestamp']}")
            print(f"   Query: {entry['query']}")
            print(f"   Answer: {entry['answer']}")
            print(f"   Tokens used: {entry['tokens_used']}")
            print(f"   Latency: {entry['latency_ms']}ms")
            print(f"   Citations count: {entry['citations_count']}")
            print(f"   Input file: {entry['input_file']}")
            print(f"   Model: {entry['model_name']}")
            print(f"   Answer length: {entry['answer_length']}")
            print(f"   Query length: {entry['query_length']}")
            
            assert entry["query"] == "What is AI?"
            assert entry["answer"] == "AI is artificial intelligence"
            assert entry["tokens_used"] == 45
            assert entry["latency_ms"] == 123.45
            assert entry["citations_count"] == 1
            assert entry["input_file"] == "test.txt"
            assert entry["model_name"] == "llama2"
            assert "timestamp" in entry
        
        print("✅ Log interaction test PASSED")
    finally:
        if os.path.exists(log_file):
            os.unlink(log_file)


def test_get_stats_empty():
    """Test getting stats from empty log file"""
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp_file:
        log_file = tmp_file.name
    
    try:
        logger = QALogger(log_file)
        stats = logger.get_stats()
        
        assert stats["total_interactions"] == 0
        assert stats["total_tokens"] == 0
        assert stats["total_cost_usd"] == 0.0
        assert stats["avg_latency_ms"] == 0.0
        
        print("✅ Empty stats test PASSED")
    finally:
        if os.path.exists(log_file):
            os.unlink(log_file)


def test_get_stats_with_data():
    """Test getting stats from log file with data"""
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp_file:
        log_file = tmp_file.name
    
    try:
        logger = QALogger(log_file)
        
        print(f"   📝 Logging multiple interactions...")
        
        # Log multiple interactions
        logger.log_interaction(
            query="Question 1",
            answer="Answer 1",
            tokens_used=50,
            latency_ms=100.0,
            citations=[],
            input_file="test.txt"
        )
        print(f"   Interaction 1: 50 tokens, 100ms")
        
        logger.log_interaction(
            query="Question 2",
            answer="Answer 2",
            tokens_used=75,
            latency_ms=200.0,
            citations=[],
            input_file="test.txt"
        )
        print(f"   Interaction 2: 75 tokens, 200ms")
        
        stats = logger.get_stats()
        
        print(f"\n   📊 Calculated statistics:")
        print(f"   Total interactions: {stats['total_interactions']}")
        print(f"   Total tokens: {stats['total_tokens']}")
        print(f"   Total cost: ${stats['total_cost_usd']}")
        print(f"   Average latency: {stats['avg_latency_ms']}ms")
        print(f"   Log file: {stats['log_file']}")
        
        assert stats["total_interactions"] == 2
        assert stats["total_tokens"] == 125
        assert stats["total_cost_usd"] == 0.0
        assert stats["avg_latency_ms"] == 150.0
        
        print("✅ Stats with data test PASSED")
    finally:
        if os.path.exists(log_file):
            os.unlink(log_file)


def test_logger_with_citations():
    """Test logging with citations"""
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp_file:
        log_file = tmp_file.name
    
    try:
        logger = QALogger(log_file)
        
        citations = [
            {
                "chunk_id": "chunk_1",
                "text": "This is a long citation text that should be truncated for preview",
                "relevance_score": 0.92
            },
            {
                "chunk_id": "chunk_2", 
                "text": "Short text",
                "relevance_score": 0.78
            }
        ]
        
        print(f"   📝 Logging with citations...")
        print(f"   Citation 1: chunk_1 (score: 0.92)")
        print(f"   Citation 2: chunk_2 (score: 0.78)")
        
        logger.log_interaction(
            query="Test question",
            answer="Test answer",
            tokens_used=60,
            latency_ms=150.0,
            citations=citations,
            input_file="test.txt"
        )
        
        with open(log_file, 'r', encoding='utf-8') as f:
            entry = json.loads(f.readline())
            print(f"\n   📊 Citation data logged:")
            print(f"   Citations count: {len(entry['citations'])}")
            print(f"   Citation 1 ID: {entry['citations'][0]['chunk_id']}")
            print(f"   Citation 1 score: {entry['citations'][0]['relevance_score']}")
            print(f"   Citation 1 preview: {entry['citations'][0]['text_preview']}")
            print(f"   Citation 2 ID: {entry['citations'][1]['chunk_id']}")
            print(f"   Citation 2 score: {entry['citations'][1]['relevance_score']}")
            print(f"   Citation 2 preview: {entry['citations'][1]['text_preview']}")
            
            assert len(entry["citations"]) == 2
            assert entry["citations"][0]["chunk_id"] == "chunk_1"
            assert entry["citations"][0]["relevance_score"] == 0.92
            assert len(entry["citations"][0]["text_preview"]) <= 103  # 100 chars + "..."
        
        print("✅ Logger with citations test PASSED")
    finally:
        if os.path.exists(log_file):
            os.unlink(log_file) 