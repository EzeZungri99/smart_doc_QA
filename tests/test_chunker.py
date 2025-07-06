import pytest
from smartqa.chunker import TextChunker, Chunk


def test_chunker_basic():
    text = (
        "Artificial intelligence is a branch of computer science.\n\n"
        "Machine learning allows computers to learn from data.\n\n"
        "Deep learning uses neural networks for complex tasks."
    )
    
    chunker = TextChunker()
    chunks = chunker.create_chunks(text)
    
    assert len(chunks) == 3, f"Expected 3 chunks, got {len(chunks)}"
    
    for i, chunk in enumerate(chunks):
        assert isinstance(chunk, Chunk), f"Chunk {i} is not a Chunk instance"
        assert chunk.id == f"chunk_{i:04d}", f"Expected chunk_{i:04d}, got {chunk.id}"
        assert len(chunk.text) > 0, f"Chunk {i} has empty text"
    
    print("✅ Basic chunker test PASSED")


def test_chunker_empty_text():
    chunker = TextChunker()
    chunks = chunker.create_chunks("")
    
    assert len(chunks) == 0, "Empty text should produce no chunks"


def test_chunker_single_paragraph():
    text = "This is a single paragraph without line breaks."
    
    chunker = TextChunker()
    chunks = chunker.create_chunks(text)
    
    assert len(chunks) == 1, "Single paragraph should produce one chunk"
    assert chunks[0].text == text, "Chunk text should match input text"


def test_chunker_multiple_empty_lines():
    text = "First paragraph.\n\n\n\nSecond paragraph.\n\n\nThird paragraph."
    
    chunker = TextChunker()
    chunks = chunker.create_chunks(text)
    
    assert len(chunks) == 3, "Should ignore multiple empty lines"
    assert chunks[0].text == "First paragraph."
    assert chunks[1].text == "Second paragraph."
    assert chunks[2].text == "Third paragraph."
