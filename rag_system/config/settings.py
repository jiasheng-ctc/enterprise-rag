"""
Simplified Enterprise RAG Configuration
"""
import os
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Centralized configuration for the enterprise RAG system"""
    
    # Model Configuration
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-ai/all-MiniLM-L6-v2")
    GENERATION_MODEL = os.getenv("GENERATION_MODEL", "qwen2:7b")  # Changed from qwen3:8b
    RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
    
    # Ollama Configuration
    OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    
    # Storage Configuration
    LANCEDB_PATH = os.getenv("LANCEDB_PATH", "./index_store/lancedb")
    SESSION_DB_PATH = os.getenv("SESSION_DB_PATH", "./backend/chat_data.db")
    
    # Processing Configuration
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
    EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "50"))
    
    # Retrieval Configuration
    RETRIEVAL_K = 50    
    RERANK_TOP_K = 20
    
    # API Configuration
    BACKEND_PORT = int(os.getenv("BACKEND_PORT", "8000"))
    FRONTEND_PORT = int(os.getenv("FRONTEND_PORT", "3000"))
    
    @classmethod
    def get_pipeline_config(cls) -> Dict[str, Any]:
        """Get simplified pipeline configuration"""
        return {
            "embedding": {
                "model": cls.EMBEDDING_MODEL,
                "batch_size": cls.EMBEDDING_BATCH_SIZE
            },
            "retrieval": {
                "type": "hybrid",
                "vector_weight": 0.7,
                "keyword_weight": 0.3,
                "top_k": cls.RETRIEVAL_K
            },
            "reranking": {
                "enabled": True,
                "model": cls.RERANKER_MODEL,
                "top_k": cls.RERANK_TOP_K
            },
            "generation": {
                "model": cls.GENERATION_MODEL,
                "temperature": 0.7,
                "streaming": True
            },
            "chunking": {
                "strategy": "recursive",
                "chunk_size": cls.CHUNK_SIZE,
                "overlap": cls.CHUNK_OVERLAP
            }
        }

config = Config()
