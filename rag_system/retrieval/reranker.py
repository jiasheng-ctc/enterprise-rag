"""
Optimized Reranker using BGE-reranker with better performance
"""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict, Any
import numpy as np
import logging
import os
from functools import lru_cache

logger = logging.getLogger(__name__)

class Reranker:
    """Optimized reranker for improving retrieval results"""
    
    _instance = None
    _model = None
    _tokenizer = None
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern to ensure only one model is loaded"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, model_name: str = "BAAI/bge-reranker-base", use_fp16: bool = True):
        # Only initialize once
        if hasattr(self, '_initialized'):
            return
            
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_fp16 = use_fp16 and self.device == "cuda"
        self._initialized = True
        
        # Don't load model immediately - wait for first use
        logger.info(f"Reranker initialized (lazy loading enabled)")
        
    def _load_model(self):
        """Lazy load the reranker model with optimizations"""
        if Reranker._model is None:
            logger.info(f"Loading reranker model: {self.model_name}")
            
            try:
                # Load tokenizer
                Reranker._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    use_fast=True  # Use fast tokenizer for better performance
                )
                
                # Load model with optimizations
                Reranker._model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.use_fp16 else torch.float32,
                    device_map="auto" if self.device == "cuda" else None
                )
                
                if self.device == "cuda" and not hasattr(Reranker._model, 'hf_device_map'):
                    Reranker._model.to(self.device)
                    
                Reranker._model.eval()
                
                # Enable torch compile for faster inference (if available)
                if hasattr(torch, 'compile') and self.device == "cuda":
                    try:
                        Reranker._model = torch.compile(Reranker._model, mode="reduce-overhead")
                        logger.info("Model compiled with torch.compile")
                    except:
                        logger.info("torch.compile not available or failed")
                
                logger.info(f"Reranker loaded on {self.device} (fp16={self.use_fp16})")
                
            except Exception as e:
                logger.error(f"Failed to load reranker model: {e}")
                # Set to False to indicate no reranker available
                Reranker._model = False
                Reranker._tokenizer = False
                raise
    
    def is_available(self) -> bool:
        """Check if reranker is available"""
        if Reranker._model is False:
            return False
        if Reranker._model is None:
            try:
                self._load_model()
            except:
                return False
        return Reranker._model is not False
    
    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """Rerank documents based on relevance to query"""
        if not documents:
            return []
        
        try:
            self._load_model()
            
            # If model failed to load, return original documents
            if self._model is None:
                logger.warning("Reranker not available, returning original order")
                return documents[:top_k]
            
            # Score each document
            scores = []
            for doc in documents:
                score = self._score_pair(query, doc["text"])
                scores.append(score)
            
            # Sort documents by score
            scored_docs = list(zip(scores, documents))
            scored_docs.sort(key=lambda x: x[0], reverse=True)
            
            # Return top_k documents with rerank scores
            results = []
            for score, doc in scored_docs[:top_k]:
                doc_copy = doc.copy()
                doc_copy["rerank_score"] = float(score)
                results.append(doc_copy)
            
            return results
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            logger.warning("Returning original document order")
            return documents[:top_k]

    def _batch_score(self, query: str, texts: List[str], batch_size: int = 8) -> List[float]:
        """Score multiple documents in batches for efficiency"""
        scores = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_scores = self._score_batch(query, batch_texts)
            scores.extend(batch_scores)
        
        return scores
    
    def _score_batch(self, query: str, texts: List[str]) -> List[float]:
        """Score a batch of query-document pairs"""
        # Prepare inputs
        inputs = Reranker._tokenizer(
            [query] * len(texts),
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get scores
        with torch.no_grad():
            if self.use_fp16:
                with torch.cuda.amp.autocast():
                    outputs = Reranker._model(**inputs)
            else:
                outputs = Reranker._model(**inputs)
            
            # Get relevance scores (assuming binary classification with relevance as positive class)
            scores = outputs.logits[:, 0].cpu().numpy().tolist()
        
        return scores
    
    def _score_pair(self, query: str, text: str) -> float:
        """Score a single query-document pair (for backwards compatibility)"""
        return self._score_batch(query, [text])[0]
    
    def preload(self):
        """Preload the model (useful for warming up)"""
        try:
            self._load_model()
            logger.info("Reranker model preloaded successfully")
        except Exception as e:
            logger.error(f"Failed to preload reranker: {e}")

# Global reranker instance (singleton)
_global_reranker = None

def get_reranker(model_name: str = "BAAI/bge-reranker-base") -> Reranker:
    """Get or create the global reranker instance"""
    global _global_reranker
    if _global_reranker is None:
        _global_reranker = Reranker(model_name)
    return _global_reranker