"""
Fixed Vector Store with proper table creation
"""
import lancedb
import pyarrow as pa
import numpy as np
from typing import List, Dict, Any, Optional
import json
import logging
from pathlib import Path
from datetime import datetime
import shutil

logger = logging.getLogger(__name__)

class VectorStore:
    """Vector store with session isolation and dynamic dimensions"""
    
    def __init__(self, db_path: str = "./index_store/lancedb", embedding_dim: Optional[int] = None):
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(str(self.db_path))
        self.embedding_dim = embedding_dim  # Will be set dynamically
        self._tables = {}  # Cache for open tables
        
    def _get_embedding_dim(self, embeddings: np.ndarray) -> int:
        """Get embedding dimension from the actual embeddings"""
        if self.embedding_dim is None:
            self.embedding_dim = embeddings.shape[1]
            logger.info(f"Set embedding dimension to {self.embedding_dim}")
        return self.embedding_dim
        
    def _get_table(self, session_id: str = "default", embedding_dim: Optional[int] = None):
        """Get or create a session-specific table with correct dimensions"""
        # Sanitize session_id for table name
        table_name = f"session_{session_id.replace('-', '_')}"
        
        # Check cache first
        if table_name in self._tables:
            return self._tables[table_name]
        
        # Check if table exists
        if table_name in self.db.table_names():
            self._tables[table_name] = self.db.open_table(table_name)
        else:
            # Use provided dimension or fallback
            dim = embedding_dim or self.embedding_dim or 384  # Default fallback
            
            # FIXED: Create table with actual data instead of empty list
            # We'll create it in add_documents method when we have real data
            logger.info(f"Will create new table for session: {table_name} with dimension {dim}")
            return None  # Return None to indicate table needs creation
        
        return self._tables[table_name]
    
    def _create_table_with_data(self, table_name: str, data: List[Dict], embedding_dim: int):
        """Create a new table with actual data"""
        try:
            # Create table with real data
            self._tables[table_name] = self.db.create_table(table_name, data=data)
            logger.info(f"Created new table: {table_name} with {len(data)} records and dimension {embedding_dim}")
            return self._tables[table_name]
        except Exception as e:
            logger.error(f"Failed to create table {table_name}: {e}")
            raise
    
    def add_documents(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray, session_id: str = "default"):
        """Add documents with their embeddings to a session-specific store"""
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks and embeddings must match")
        
        if len(embeddings) == 0:
            logger.warning("No embeddings to add")
            return
        
        # Set embedding dimension from actual data
        embedding_dim = self._get_embedding_dim(embeddings)
        
        # Sanitize session_id for table name
        table_name = f"session_{session_id.replace('-', '_')}"
        
        # Get existing table or prepare to create new one
        table = self._get_table(session_id, embedding_dim)
        
        timestamp = datetime.now().isoformat()
        
        # Prepare data
        data = []
        for chunk, embedding in zip(chunks, embeddings):
            data.append({
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "vector": embedding.tolist(),
                "document_id": chunk["metadata"]["document_id"],
                "metadata": json.dumps(chunk["metadata"]),
                "session_id": session_id,
                "timestamp": timestamp
            })
        
        if table is None:
            # Create new table with data
            table = self._create_table_with_data(table_name, data, embedding_dim)
        else:
            # Add to existing table
            table.add(data)
            
        logger.info(f"Added {len(data)} chunks to session {session_id}")
    
    def search(self, query_embedding: np.ndarray, k: int = 5, session_id: str = "default") -> List[Dict[str, Any]]:
        """Search for similar documents within a session"""
        try:
            table = self._get_table(session_id)
            
            if table is None:
                logger.info(f"No table exists for session {session_id}")
                return []
            
            # Check if table has data
            try:
                df = table.to_pandas()
                if df.empty:
                    logger.info(f"No documents in session {session_id}")
                    return []
            except Exception as e:
                logger.warning(f"Could not check table contents: {e}")
                return []
            
            results = table.search(query_embedding.tolist()).limit(k).to_list()
            
            # Parse metadata and format results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "chunk_id": result["chunk_id"],
                    "text": result["text"],
                    "document_id": result["document_id"],
                    "score": result.get("_distance", 0),
                    "metadata": json.loads(result["metadata"])
                })
            
            return formatted_results
        except Exception as e:
            logger.error(f"Search error in session {session_id}: {e}")
            return []
    
    def hybrid_search(self, query_embedding: np.ndarray, query_text: str, k: int = 5, session_id: str = "default") -> List[Dict[str, Any]]:
        """Perform hybrid search within a session"""
        try:
            table = self._get_table(session_id)
            
            if table is None:
                logger.info(f"No table exists for session {session_id}")
                return []
            
            # Check if table has data
            try:
                df = table.to_pandas()
                if df.empty:
                    logger.info(f"No documents in session {session_id}")
                    return []
            except Exception as e:
                logger.warning(f"Could not check table contents: {e}")
                return []
            
            # Vector search
            vector_results = table.search(query_embedding.tolist()).limit(k * 2).to_list()
            
            # Simple keyword filtering
            keyword_filtered = []
            query_terms = query_text.lower().split()
            
            for result in vector_results:
                text_lower = result["text"].lower()
                if any(term in text_lower for term in query_terms):
                    result["hybrid_score"] = result.get("_distance", 0) * 0.7 + 0.3
                else:
                    result["hybrid_score"] = result.get("_distance", 0) * 0.7
                keyword_filtered.append(result)
            
            # Sort by hybrid score and return top k
            keyword_filtered.sort(key=lambda x: x["hybrid_score"])
            
            formatted_results = []
            for result in keyword_filtered[:k]:
                formatted_results.append({
                    "chunk_id": result["chunk_id"],
                    "text": result["text"],
                    "document_id": result["document_id"],
                    "score": result["hybrid_score"],
                    "metadata": json.loads(result["metadata"])
                })
            
            return formatted_results
        except Exception as e:
            logger.error(f"Hybrid search error in session {session_id}: {e}")
            return []
    
    def cleanup_session(self, session_id: str):
        """Delete all data for a session"""
        table_name = f"session_{session_id.replace('-', '_')}"
        
        try:
            # Remove from cache
            if table_name in self._tables:
                del self._tables[table_name]
            
            # Drop table if exists
            if table_name in self.db.table_names():
                # LanceDB doesn't have direct drop_table, so we'll use a workaround
                # Delete the table directory
                table_path = self.db_path / table_name
                if table_path.exists():
                    shutil.rmtree(table_path)
                logger.info(f"Cleaned up session table: {table_name}")
            
        except Exception as e:
            logger.error(f"Error cleaning up session {session_id}: {e}")
    
    def cleanup_old_sessions(self, hours: int = 24):
        """Clean up sessions older than specified hours"""
        from datetime import datetime, timedelta
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        for table_name in self.db.table_names():
            if table_name.startswith("session_"):
                try:
                    table = self.db.open_table(table_name)
                    df = table.to_pandas()
                    
                    if not df.empty:
                        # Check the timestamp of the first record
                        first_timestamp = df.iloc[0]['timestamp']
                        if isinstance(first_timestamp, str):
                            timestamp = datetime.fromisoformat(first_timestamp)
                            if timestamp < cutoff_time:
                                session_id = table_name.replace("session_", "").replace("_", "-")
                                self.cleanup_session(session_id)
                                logger.info(f"Cleaned up old session: {session_id}")
                except Exception as e:
                    logger.error(f"Error checking table {table_name}: {e}")
    
    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """Get statistics for a session"""
        table_name = f"session_{session_id.replace('-', '_')}"
        
        if table_name not in self.db.table_names():
            return {"exists": False}
        
        try:
            table = self._get_table(session_id)
            if table is None:
                return {"exists": False}
                
            df = table.to_pandas()
            
            return {
                "exists": True,
                "document_count": df['document_id'].nunique() if not df.empty else 0,
                "chunk_count": len(df),
                "session_id": session_id
            }
        except Exception as e:
            logger.error(f"Error getting stats for session {session_id}: {e}")
            return {"exists": False, "error": str(e)}