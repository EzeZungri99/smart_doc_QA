import pytest
import numpy as np
from smartqa.chunker import TextChunker
from smartqa.embedder import Embedder


def test_embedder_with_chunker():
    text = (
        "Artificial intelligence is a branch of computer science.\n\n"
        "It allows machines to learn from experience."
    )
    
    chunker = TextChunker()
    chunks = chunker.create_chunks(text)
    assert len(chunks) > 0, "No chunks were generated"
    
    embedder = Embedder()
    texts = [chunk.text for chunk in chunks]
    embeddings = embedder.encode_texts(texts)
    
    assert embeddings.shape[0] == len(chunks), "Number of embeddings doesn't match chunks"
    assert embeddings.shape[1] == embedder.dimension, "Embedding dimension is not as expected"
    assert embeddings.dtype == np.float32, "Embeddings should be float32"
    
    print("✅ Embedder + chunker test PASSED")


def test_embedder_single_text():
    embedder = Embedder()
    text = "This is a test sentence."
    
    embedding = embedder.encode_single(text)
    
    assert embedding.shape[0] == 1, "Single text should produce one embedding"
    assert embedding.shape[1] == embedder.dimension, "Embedding dimension should match model"
    assert embedding.dtype == np.float32, "Embedding should be float32"


def test_embedder_multiple_texts():
    embedder = Embedder()
    texts = [
        "First sentence for testing.",
        "Second sentence for testing.",
        "Third sentence for testing."
    ]
    
    embeddings = embedder.encode_texts(texts)
    
    assert embeddings.shape[0] == len(texts), "Number of embeddings should match number of texts"
    assert embeddings.shape[1] == embedder.dimension, "Embedding dimension should match model"
    assert embeddings.dtype == np.float32, "Embeddings should be float32"


def test_embedder_empty_text():
    embedder = Embedder()
    
    embedding = embedder.encode_single("")
    assert embedding.shape[0] == 1, "Empty text should still produce one embedding"
    
    embeddings = embedder.encode_texts(["", "", ""])
    assert embeddings.shape[0] == 3, "Should handle multiple empty texts"


def test_embedder_model_dimension():
    embedder = Embedder()
    
    assert hasattr(embedder, 'dimension'), "Embedder should have dimension attribute"
    assert embedder.dimension > 0, "Dimension should be positive"
    assert isinstance(embedder.dimension, int), "Dimension should be integer" 