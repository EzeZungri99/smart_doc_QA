from smartqa.chunker import TextChunker
from smartqa.embedder import Embedder
from smartqa.retriever import Retriever

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
    
    print("Retriever + chunker + embedder test PASSED successfully.") 