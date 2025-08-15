"""
Simplified RAG Pipeline with Session Support
"""
from typing import List, Dict, Any, Optional
import logging
from rag_system.processing.document_processor import DocumentProcessor
from rag_system.processing.chunker import RecursiveChunker
from rag_system.retrieval.embedder import EmbeddingManager
from rag_system.retrieval.reranker import Reranker
from rag_system.storage.vector_store import VectorStore
from rag_system.storage.session_manager import SessionManager
from rag_system.config.settings import config
from pathlib import Path
import shutil
import sqlite3
import json  
from datetime import datetime  

logger = logging.getLogger(__name__)

class RAGPipeline:
    """Main RAG pipeline orchestrator with session support"""
        
    def __init__(self):
            self.config = config.get_pipeline_config()
            self.document_processor = DocumentProcessor()
            self.chunker = RecursiveChunker(
                chunk_size=self.config["chunking"]["chunk_size"],
                chunk_overlap=self.config["chunking"]["overlap"]
            )
            self.embedder = EmbeddingManager(
                model_name=self.config["embedding"]["model"],
                batch_size=self.config["embedding"]["batch_size"]
            )
            # Don't set embedding_dim yet - let it be determined dynamically
            self.vector_store = VectorStore(db_path=config.LANCEDB_PATH)
            self.reranker = Reranker(model_name=self.config["reranking"]["model"])
            self.session_manager = SessionManager(db_path=config.SESSION_DB_PATH)
            
            # Track active sessions and their resources
            self.active_sessions = {}
            
    def index_documents(self, file_paths: List[str], session_id: Optional[str] = None) -> Dict[str, Any]:
        """Index a batch of documents"""
        logger.info(f"Indexing {len(file_paths)} documents for session {session_id}")
        
        try:
            # Process documents
            documents = self.document_processor.process_batch(file_paths)
            if not documents:
                logger.warning("No valid documents to process")
                return {"status": "error", "message": "No valid documents to process"}
            
            logger.info(f"Successfully processed {len(documents)} documents")
            
            # Chunk documents
            all_chunks = []
            for doc in documents:
                chunks = self.chunker.chunk_text(doc["content"], doc["document_id"])
                all_chunks.extend(chunks)
                logger.info(f"Document '{doc['filename']}' created {len(chunks)} chunks")
            
            logger.info(f"Created {len(all_chunks)} chunks total")
            
            if not all_chunks:
                logger.warning("No chunks created from documents")
                return {"status": "error", "message": "No chunks created from documents"}
            
            # Generate embeddings
            texts = [chunk["text"] for chunk in all_chunks]
            logger.info(f"Generating embeddings for {len(texts)} chunks...")
            embeddings = self.embedder.embed_texts(texts)
            logger.info(f"Generated embeddings with shape: {embeddings.shape}")
            
            # Store in vector database
            if session_id:
                self.vector_store.add_documents(all_chunks, embeddings, session_id=session_id)
            else:
                self.vector_store.add_documents(all_chunks, embeddings)
            
            logger.info(f"Stored {len(all_chunks)} chunks in vector database")
            
            # IMPORTANT: Update session metadata with correct counts
            if session_id:
                import json
                conn = sqlite3.connect(str(self.session_manager.db_path))
                cursor = conn.cursor()
                
                # Get current metadata
                cursor.execute('SELECT metadata FROM sessions WHERE id = ?', (session_id,))
                result = cursor.fetchone()
                metadata = json.loads(result[0]) if result and result[0] else {}
                
                # Update with cumulative counts
                existing_docs = metadata.get('document_count', 0)
                existing_chunks = metadata.get('chunk_count', 0)
                
                new_metadata = {
                    'document_count': existing_docs + len(documents),
                    'chunk_count': existing_chunks + len(all_chunks),
                    'last_indexed': datetime.now().isoformat()
                }
                metadata.update(new_metadata)
                
                cursor.execute(
                    'UPDATE sessions SET metadata = ? WHERE id = ?',
                    (json.dumps(metadata), session_id)
                )
                conn.commit()
                conn.close()
                
                logger.info(f"Updated session metadata - Total docs: {metadata['document_count']}, Total chunks: {metadata['chunk_count']}")
            
            return {
                "status": "success",
                "documents_processed": len(documents),
                "chunks_created": len(all_chunks),
                "session_id": session_id
            }
            
        except Exception as e:
            logger.error(f"Error in index_documents: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"status": "error", "message": str(e)}


    # def retrieve(self, query: str, session_id: str = "default", k: int = 20, use_reranker: bool = True) -> List[Dict[str, Any]]:
    #     """Retrieve relevant documents from a specific session"""
    #     # Embed query
    #     query_embedding = self.embedder.embed_query(query)
        
    #     # Hybrid search within session
    #     results = self.vector_store.hybrid_search(
    #         query_embedding, 
    #         query, 
    #         k=k,
    #         session_id=session_id
    #     )
        
    #     # Rerank if enabled and results exist
    #     if use_reranker and results:
    #         results = self.reranker.rerank(
    #             query, 
    #             results, 
    #             top_k=self.config["reranking"]["top_k"]
    #         )
        
    #     return results

    def retrieve(self, query: str, session_id: str = "default", k: int = 50, use_reranker: bool = True) -> List[Dict[str, Any]]:
        """Retrieve relevant documents from a specific session with balanced multi-document support"""
        try:
            # Embed query
            query_embedding = self.embedder.embed_query(query)
            
            # Hybrid search within session - get more initial results
            initial_k = min(k * 2, 100)
            results = self.vector_store.hybrid_search(
                query_embedding, 
                query, 
                k=initial_k,
                session_id=session_id
            )
            
            if not results:
                return []
            
            # Rerank if enabled
            if use_reranker and results:
                # Increase the reranking limit
                rerank_k = min(k, self.config["reranking"]["top_k"])
                results = self.reranker.rerank(
                    query, 
                    results[:k],  # Limit before reranking
                    top_k=rerank_k
                )
            
            return results[:k]
            
        except Exception as e:
            logger.error(f"Error in retrieve: {e}")
            return []

    def query(self, query: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Process a query and generate response for a session"""
        # Create session if not provided
        if not session_id:
            session_id = self.session_manager.create_session()
        
        # Add user message
        self.session_manager.add_message(session_id, "user", query)
        
        # Retrieve relevant documents from this session only
        documents = self.retrieve(query, session_id=session_id)
        
        # Generate response
        response = self._generate_response(query, documents)
        
        # Add assistant message
        self.session_manager.add_message(session_id, "assistant", response["answer"])
        
        return {
            "session_id": session_id,
            "answer": response["answer"],
            "source_documents": documents[:5]  # Return top 5 sources
        }
    
    def cleanup_session(self, session_id: str):
        """Clean up all resources for a session"""
        logger.info(f"Cleaning up session: {session_id}")
        
        try:
            # 1. Clean up vector store
            self.vector_store.cleanup_session(session_id)
            
            # 2. Clean up uploaded files
            if session_id in self.active_sessions:
                # Remove files
                for file_path in self.active_sessions[session_id].get("file_paths", []):
                    try:
                        Path(file_path).unlink()
                        logger.info(f"Deleted file: {file_path}")
                    except Exception as e:
                        logger.error(f"Error deleting file {file_path}: {e}")
                
                # Remove session from tracking
                del self.active_sessions[session_id]
            
            # 3. Also check shared_uploads directory
            upload_dir = Path("./shared_uploads")
            if upload_dir.exists():
                # Remove session-specific directory
                session_dir = upload_dir / session_id
                if session_dir.exists():
                    shutil.rmtree(session_dir)
                    logger.info(f"Deleted session directory: {session_dir}")
                
                # Remove any files with session_id prefix
                for file in upload_dir.glob(f"{session_id}_*"):
                    file.unlink()
                    logger.info(f"Deleted file: {file}")
                
                for file in upload_dir.glob(f"*_{session_id}_*"):
                    file.unlink()
                    logger.info(f"Deleted file: {file}")
            
            logger.info(f"Successfully cleaned up session: {session_id}")
            
        except Exception as e:
            logger.error(f"Error during session cleanup: {e}")
            raise
    
    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """Get statistics for a session"""
        stats = self.vector_store.get_session_stats(session_id)
        
        if session_id in self.active_sessions:
            stats["active"] = True
            stats["file_count"] = len(self.active_sessions[session_id].get("file_paths", []))
            stats["document_ids"] = self.active_sessions[session_id].get("documents", [])
        else:
            stats["active"] = False
            stats["file_count"] = 0
            stats["document_ids"] = []
        
        return stats
    
    def cleanup_old_sessions(self, hours: int = 24):
        """Clean up sessions older than specified hours"""
        logger.info(f"Cleaning up sessions older than {hours} hours")
        self.vector_store.cleanup_old_sessions(hours)
    
    def _generate_response(self, query: str, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate response using retrieved documents"""
        if not documents:
            return {"answer": "I couldn't find relevant information in your uploaded documents to answer this question. Please make sure you've uploaded relevant documents first."}
        
        context = "\n\n".join([doc["text"] for doc in documents[:5]])
        
        # This will be replaced by actual Ollama generation in the server
        answer = f"Based on the retrieved documents, here's what I found:\n\n{context[:500]}..."
        
        return {"answer": answer}