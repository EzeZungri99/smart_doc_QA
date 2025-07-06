import pytest
from smartqa.chunker import TextChunker
from smartqa.embedder import Embedder
from smartqa.retriever import Retriever
from smartqa.llm import LLMResponseGenerator


def load_example_text():
    try:
        with open("example.txt", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "Machine learning allows computers to learn from data without being explicitly programmed."


def test_llm_basic():
    text = load_example_text()
    
    chunker = TextChunker()
    chunks = chunker.create_chunks(text)
    
    embedder = Embedder()
    retriever = Retriever(embedder)
    retriever.add_chunks(chunks)
    
    query = "What is artificial intelligence?"
    search_results = retriever.search(query, k=2)
    
    llm = LLMResponseGenerator()
    response = llm.generate_response(query, search_results, "example.txt")
    
    assert "answer" in response, "Response missing answer field"
    assert "citations" in response, "Response missing citations field"
    assert "tokens_used" in response, "Response missing tokens_used field"
    assert "latency_ms" in response, "Response missing latency_ms field"
    
    assert isinstance(response["answer"], str), "Answer should be a string"
    assert isinstance(response["citations"], list), "Citations should be a list"
    assert isinstance(response["tokens_used"], int), "Tokens used should be an integer"
    assert isinstance(response["latency_ms"], (int, float)), "Latency should be a number"
    
    print("✅ LLM basic test PASSED")


def test_llm_without_relevant_answer():
    text = load_example_text()
    
    chunker = TextChunker()
    chunks = chunker.create_chunks(text)
    
    embedder = Embedder()
    retriever = Retriever(embedder)
    retriever.add_chunks(chunks)
    
    query = "What is quantum physics?"
    search_results = retriever.search(query, k=2)
    
    llm = LLMResponseGenerator()
    response = llm.generate_response(query, search_results, "example.txt")
    
    assert "answer" in response, "Response missing answer field"
    assert "citations" in response, "Response missing citations field"
    assert "tokens_used" in response, "Response missing tokens_used field"
    
    print("✅ LLM test without relevant answer PASSED")


def test_llm_with_completely_unrelated_question():
    text = load_example_text()
    
    chunker = TextChunker()
    chunks = chunker.create_chunks(text)
    
    embedder = Embedder()
    retriever = Retriever(embedder)
    retriever.add_chunks(chunks)
    
    query = "Do you like bananas?"
    search_results = retriever.search(query, k=2)
    
    llm = LLMResponseGenerator()
    response = llm.generate_response(query, search_results, "example.txt")
    
    assert "answer" in response, "Response missing answer field"
    assert "citations" in response, "Response missing citations field"
    assert "tokens_used" in response, "Response missing tokens_used field"
    
    answer_lower = response["answer"].lower()
    assert any(phrase in answer_lower for phrase in [
        "don't know", "not know", "no information", "not familiar", 
        "not mentioned", "not covered", "not found", "not available",
        "i don't have information"
    ]), f"Answer should indicate lack of knowledge, got: {response['answer']}"
    
    print("✅ LLM test with unrelated question PASSED")


def test_llm_with_empty_chunks():
    llm = LLMResponseGenerator()
    response = llm.generate_response("test question", [], "test.txt")
    
    assert response["answer"] == "I don't have information about this topic in the provided document."
    assert response["citations"] == []
    assert response["tokens_used"] == 0
    assert response["latency_ms"] == 0
    
    print("✅ LLM test with empty chunks PASSED")


def test_llm_response_structure():
    text = load_example_text()
    
    chunker = TextChunker()
    chunks = chunker.create_chunks(text)
    
    embedder = Embedder()
    retriever = Retriever(embedder)
    retriever.add_chunks(chunks)
    
    query = "What is artificial intelligence?"
    search_results = retriever.search(query, k=2)
    
    llm = LLMResponseGenerator()
    response = llm.generate_response(query, search_results, "example.txt")
    
    required_fields = ["answer", "citations", "tokens_used", "latency_ms"]
    for field in required_fields:
        assert field in response, f"Response missing {field} field"
    
    assert isinstance(response["answer"], str), "Answer should be a string"
    assert isinstance(response["citations"], list), "Citations should be a list"
    assert isinstance(response["tokens_used"], int), "Tokens used should be an integer"
    assert isinstance(response["latency_ms"], (int, float)), "Latency should be a number"
    assert response["tokens_used"] >= 0, "Tokens used should be non-negative"
    assert response["latency_ms"] >= 0, "Latency should be non-negative"
    
    print("✅ LLM response structure test PASSED") 