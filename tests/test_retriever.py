import pytest
from smartqa.chunker import TextChunker, Chunk
from smartqa.embedder import Embedder
from smartqa.retriever import Retriever


def load_example_text():
    return (
        "Artificial intelligence is a branch of computer science.\n\n"
        "Machine learning allows computers to learn from data.\n\n"
        "Deep learning uses neural networks for complex tasks."
    )


def test_retriever_with_chunker_and_embedder():
    text = (
        "Artificial intelligence is a branch of computer science.\n\n"
        "Machine learning allows computers to learn from data.\n\n"
        "Deep learning uses neural networks for complex tasks."
    )
    
    chunker = TextChunker()
    chunks = chunker.create_chunks(text)
    assert len(chunks) > 0, "No chunks were generated"
    
    embedder = Embedder()
    retriever = Retriever(embedder)
    retriever.add_chunks(chunks)
    
    query = "machine learning"
    results = retriever.search(query, k=2)
    
    assert len(results) > 0, "No search results found"
    assert len(results) <= 2, "Too many results returned"
    
    for chunk, score in results:
        assert isinstance(chunk.text, str), "Chunk text is not a string"
        assert isinstance(score, float), "Score is not a float"
        assert 0 <= score <= 1, "Score is not in valid range"
    
    print("✅ Retriever + chunker + embedder test PASSED")


def test_retriever_with_unrelated_query():
    text = load_example_text()
    
    chunker = TextChunker()
    chunks = chunker.create_chunks(text)
    
    embedder = Embedder()
    retriever = Retriever(embedder)
    retriever.add_chunks(chunks)
    
    query = "What is the weather like today?"
    results = retriever.search(query, k=3)
    
    assert isinstance(results, list), "Results should be a list"
    assert len(results) <= 3, "Should return at most 3 results"
    
    if results:
        for chunk, score in results:
            assert isinstance(chunk, Chunk), "Each result should contain a Chunk"
            assert isinstance(score, (int, float)), "Each result should contain a score"
    
    print("✅ Retriever test with unrelated query PASSED")


def test_retriever_with_personal_question():
    text = load_example_text()
    
    chunker = TextChunker()
    chunks = chunker.create_chunks(text)
    
    embedder = Embedder()
    retriever = Retriever(embedder)
    retriever.add_chunks(chunks)
    
    query = "What is your favorite movie?"
    results = retriever.search(query, k=3)
    
    assert isinstance(results, list), "Results should be a list"
    assert len(results) <= 3, "Should return at most 3 results"
    
    print("✅ Retriever test with personal question PASSED")
    print(f"   Results found: {len(results)}")


def test_retriever_with_very_specific_technical_query():
    text = load_example_text()
    
    chunker = TextChunker()
    chunks = chunker.create_chunks(text)
    
    embedder = Embedder()
    retriever = Retriever(embedder)
    retriever.add_chunks(chunks)
    
    query = "What is the exact implementation of a transformer architecture with 12 layers and 768 hidden dimensions?"
    results = retriever.search(query, k=3)
    
    assert isinstance(results, list), "Results should be a list"
    assert len(results) <= 3, "Should return at most 3 results"
    
    print("✅ Retriever test with specific technical query PASSED")
    print(f"   Results found: {len(results)}")


def test_retriever_with_empty_chunks():
    embedder = Embedder()
    retriever = Retriever(embedder)
    
    query = "What is artificial intelligence?"
    results = retriever.search(query, k=3)
    
    assert isinstance(results, list), "Results should be a list"
    assert len(results) == 0, "Should return empty list when no chunks are available"
    
    print("✅ Retriever test with empty chunks PASSED")


def test_retriever_with_special_characters():
    text = load_example_text()
    
    chunker = TextChunker()
    chunks = chunker.create_chunks(text)
    
    embedder = Embedder()
    retriever = Retriever(embedder)
    retriever.add_chunks(chunks)
    
    query = "What is AI? (artificial intelligence) - please explain!"
    results = retriever.search(query, k=3)
    
    assert isinstance(results, list), "Results should be a list"
    assert len(results) <= 3, "Should return at most 3 results"
    
    print("✅ Retriever test with special characters PASSED")
    print(f"   Results found: {len(results)}")


def test_retriever_score_distribution():
    text = load_example_text()
    
    chunker = TextChunker()
    chunks = chunker.create_chunks(text)
    
    embedder = Embedder()
    retriever = Retriever(embedder)
    retriever.add_chunks(chunks)
    
    query = "What is artificial intelligence?"
    results = retriever.search(query, k=5)
    
    assert isinstance(results, list), "Results should be a list"
    assert len(results) <= 5, "Should return at most 5 results"
    
    if len(results) > 1:
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True), "Results should be sorted by score (descending)"
        
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], f"Score at position {i} should be >= score at position {i + 1}"
    
    print("✅ Retriever test score distribution PASSED")


def test_retriever_add_chunks():
    embedder = Embedder()
    retriever = Retriever(embedder)
    
    retriever.add_chunks([])
    assert len(retriever.chunks) == 0, "Should handle empty chunks list"
    
    chunker = TextChunker()
    chunks = chunker.create_chunks("Test text.")
    retriever.add_chunks(chunks)
    
    assert len(retriever.chunks) == len(chunks), "Should add all chunks"
    assert retriever.index is not None, "Should create index when adding chunks" 