from smartqa.chunker import TextChunker
from smartqa.embedder import Embedder

def test_embedder_with_chunker():
    text = (
        "Artificial intelligence is a branch of computer science.\n\n"
        "It allows machines to learn from experience."
    )
    # 1. Chunking
    chunker = TextChunker()
    chunks = chunker.create_chunks(text)
    assert len(chunks) > 0, "No chunks were generated"
    
    # 2. Embedding
    embedder = Embedder()
    texts = [chunk.text for chunk in chunks]
    embeddings = embedder.encode_texts(texts)
    assert embeddings.shape[0] == len(chunks), "Number of embeddings doesn't match chunks"
    assert embeddings.shape[1] == embedder.dimension, "Embedding dimension is not as expected"
    print("Embedder + chunker test PASSED successfully.") 