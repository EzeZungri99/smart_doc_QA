from typing import List

class Chunk:
    def __init__(self, id: str, text: str):
        self.id = id
        self.text = text

class TextChunker:
    
    def __init__(self, chunk_size: int = 1000):
        self.chunk_size = chunk_size
    
    def create_chunks(self, text: str) -> List[Chunk]:
        chunks = []
        chunk_id = 0
        
        paragraphs = text.split('\n\n')
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if paragraph:
                chunks.append(Chunk(
                    id=f"chunk_{chunk_id:04d}",
                    text=paragraph
                ))
                chunk_id += 1
        
        return chunks