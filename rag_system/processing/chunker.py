"""
Simplified Recursive Chunker
"""
from typing import List, Dict, Any
import re

class RecursiveChunker:
    """Simple recursive text chunker with overlap"""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = ["\n\n", "\n", ". ", " "]
        
    def chunk_text(self, text: str, document_id: str) -> List[Dict[str, Any]]:
        """Split text into chunks with metadata"""
        chunks = self._recursive_split(text)
        
        result = []
        for i, chunk_text in enumerate(chunks):
            if chunk_text.strip():
                result.append({
                    "chunk_id": f"{document_id}_chunk_{i}",
                    "text": chunk_text.strip(),
                    "metadata": {
                        "document_id": document_id,
                        "chunk_index": i,
                        "chunk_size": len(chunk_text)
                    }
                })
        
        return result
    
    def _recursive_split(self, text: str) -> List[str]:
        """Recursively split text using separators"""
        chunks = []
        current_chunk = ""
        
        # Try to split by paragraphs first
        paragraphs = text.split("\n\n")
        
        for para in paragraphs:
            if len(current_chunk) + len(para) <= self.chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # If paragraph is too large, split it further
                if len(para) > self.chunk_size:
                    sub_chunks = self._split_large_text(para)
                    chunks.extend(sub_chunks[:-1])
                    current_chunk = sub_chunks[-1] if sub_chunks else ""
                else:
                    current_chunk = para
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return self._apply_overlap(chunks)
    
    def _split_large_text(self, text: str) -> List[str]:
        """Split large text that exceeds chunk size"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0
        
        for word in words:
            word_size = len(word) + 1  # +1 for space
            if current_size + word_size <= self.chunk_size:
                current_chunk.append(word)
                current_size += word_size
            else:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_size = word_size
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def _apply_overlap(self, chunks: List[str]) -> List[str]:
        """Apply overlap between chunks"""
        if not chunks or self.chunk_overlap == 0:
            return chunks
        
        overlapped_chunks = []
        for i, chunk in enumerate(chunks):
            if i > 0 and i < len(chunks):
                # Add overlap from previous chunk
                prev_words = chunks[i-1].split()[-self.chunk_overlap:]
                chunk = " ".join(prev_words) + " " + chunk
            overlapped_chunks.append(chunk)
        
        return overlapped_chunks
