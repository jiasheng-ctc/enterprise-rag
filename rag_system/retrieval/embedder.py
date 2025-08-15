"""
Simplified Embedding Manager
"""
import numpy as np
from typing import List, Optional
import torch
from transformers import AutoTokenizer, AutoModel
import logging

logger = logging.getLogger(__name__)

class EmbeddingManager:
    """Manages text embeddings using HuggingFace models"""
    
    # Class variables to share across instances
    _model = None
    _tokenizer = None
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", batch_size: int = 32):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def _load_model(self):
        """Lazy load the model"""
        if EmbeddingManager._model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            EmbeddingManager._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            EmbeddingManager._model = AutoModel.from_pretrained(self.model_name)
            EmbeddingManager._model.to(self.device)
            EmbeddingManager._model.eval()
            logger.info(f"Model loaded on {self.device}")
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        self._load_model()
        
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = self._embed_batch(batch)
            embeddings.extend(batch_embeddings)
        
        return np.array(embeddings)
    
    def _embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Embed a batch of texts"""
        inputs = EmbeddingManager._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = EmbeddingManager._model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
        
        return embeddings.cpu().numpy()
    
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query"""
        return self.embed_texts([query])[0]