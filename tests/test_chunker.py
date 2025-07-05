from smartqa.chunker import TextChunker

def test_chunker_basic():
    text = (
        "Artificial intelligence is a branch of computer science.\n\n"
        "Machine learning allows computers to learn from data.\n\n"
        "Deep learning uses neural networks for complex tasks."
    )
    
    chunker = TextChunker()
    chunks = chunker.create_chunks(text)
    
    assert len(chunks) > 0, "No chunks were generated"
    assert len(chunks) == 3, f"Expected 3 chunks, got {len(chunks)}"
    
    for chunk in chunks:
        assert hasattr(chunk, 'id'), "Chunk missing id attribute"
        assert hasattr(chunk, 'text'), "Chunk missing text attribute"
        assert isinstance(chunk.text, str), "Chunk text is not a string"
        assert len(chunk.text) > 0, "Chunk text is empty"
    
    print("✅ Chunker test PASSED")
    print(f"   Created {len(chunks)} chunks")
