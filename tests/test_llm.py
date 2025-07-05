from smartqa.chunker import TextChunker
from smartqa.embedder import Embedder
from smartqa.retriever import Retriever
from smartqa.llm import LLMResponseGenerator


def test_llm_with_relevant_answer():
    text = "Machine learning allows computers to learn from data without being explicitly programmed."
    
    chunker = TextChunker()
    chunks = chunker.create_chunks(text)
    
    embedder = Embedder()
    retriever = Retriever(embedder)
    retriever.add_chunks(chunks)
    
    query = "What is machine learning?"
    search_results = retriever.search(query, k=1)
    
    llm = LLMResponseGenerator()
    response = llm.generate_response(query, search_results)
    
    assert "answer" in response, "Response missing answer field"
    assert "citations" in response, "Response missing citations field"
    assert "tokens_used" in response, "Response missing tokens_used field"
    assert len(response["citations"]) > 0, "Should have citations when answer found"
    assert response["tokens_used"] > 0, "Should use tokens when generating answer"
    
    print("✅ LLM test with relevant answer PASSED")
    print(f"   Answer: {response['answer'][:100]}...")
    print(f"   Tokens: {response['tokens_used']}")


def test_llm_without_relevant_answer():
    text = "Machine learning allows computers to learn from data without being explicitly programmed."
    
    chunker = TextChunker()
    chunks = chunker.create_chunks(text)
    
    embedder = Embedder()
    retriever = Retriever(embedder)
    retriever.add_chunks(chunks)
    
    query = "What is quantum physics?"
    search_results = retriever.search(query, k=1)
    
    llm = LLMResponseGenerator()
    response = llm.generate_response(query, search_results)
    
    assert "answer" in response, "Response missing answer field"
    assert "citations" in response, "Response missing citations field"
    assert "tokens_used" in response, "Response missing tokens_used field"
    
    print("✅ LLM test without relevant answer PASSED")
    print(f"   Answer: {response['answer'][:100]}...")
    print(f"   Tokens: {response['tokens_used']}")


def test_llm_with_empty_chunks():
    llm = LLMResponseGenerator()
    response = llm.generate_response("test question", [])
    
    assert response["answer"] == "No relevant information found to answer your question."
    assert response["citations"] == []
    assert response["tokens_used"] == 0
    
    print("✅ LLM test with empty chunks PASSED") 