# """
# Enhanced FastAPI Server with MCP Tools Integration
# Path: rag_system/api/server.py
# """
# from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import StreamingResponse
# from pydantic import BaseModel
# from typing import List, Optional, Dict, Any
# import json
# import sqlite3
# import logging
# import uvicorn
# from pathlib import Path
# import shutil
# import uuid
# import asyncio
# from datetime import datetime, timedelta

# from rag_system.core.rag_pipeline import RAGPipeline
# from rag_system.utils.ollama_client import OllamaClient
# from rag_system.utils.mcp_tools import MCPToolsOrchestrator, enhance_query_with_mcp_tools
# from rag_system.config.settings import config

# # Setup logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Initialize FastAPI app
# app = FastAPI(title="Enterprise RAG API with MCP Tools", version="3.0.0")

# # Configure CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Initialize components
# rag_pipeline = RAGPipeline()
# ollama_client = OllamaClient(host=config.OLLAMA_HOST)
# mcp_orchestrator = MCPToolsOrchestrator()

# # Session tracking
# active_sessions = {}
# MAX_CONCURRENT_USERS = 5
# SESSION_TIMEOUT_MINUTES = 30

# # Request/Response models
# class QueryRequest(BaseModel):
#     query: str
#     session_id: Optional[str] = None
#     retrieval_k: Optional[int] = 50  # INCREASED from 20
#     use_reranker: Optional[bool] = True
#     use_sequential_thinking: Optional[bool] = False
#     use_web_search: Optional[bool] = False

# class IndexRequest(BaseModel):
#     file_paths: List[str]
#     session_id: str
#     index_name: Optional[str] = "default"

# class ChatMessage(BaseModel):
#     role: str
#     content: str

# class ChatRequest(BaseModel):
#     messages: List[ChatMessage]
#     session_id: Optional[str] = None
#     model: Optional[str] = "qwen3:8b"
#     use_sequential_thinking: Optional[bool] = False
#     use_web_search: Optional[bool] = False

# class SessionResponse(BaseModel):
#     session_id: str
#     title: str
#     created_at: str
#     updated_at: str

# # ADDED: Function to ensure diverse document retrieval
# def get_balanced_documents(rag_pipeline, query: str, session_id: str, k: int = 50, use_reranker: bool = True) -> List[Dict[str, Any]]:
#     """Retrieve documents ensuring balanced representation from all uploaded documents"""
#     try:
#         # Get more documents initially
#         initial_k = min(k * 2, 100)
#         documents = rag_pipeline.retrieve(
#             query=query,
#             session_id=session_id,
#             k=initial_k,
#             use_reranker=use_reranker
#         )
        
#         if not documents:
#             return []
        
#         # Group documents by document_id to ensure diversity
#         document_groups = {}
#         for doc in documents:
#             doc_id = doc.get("document_id", "unknown")
#             if doc_id not in document_groups:
#                 document_groups[doc_id] = []
#             document_groups[doc_id].append(doc)
        
#         logger.info(f"Retrieved chunks from {len(document_groups)} unique documents")
        
#         # If we have multiple documents, ensure balanced representation
#         if len(document_groups) > 1:
#             balanced_results = []
#             chunks_per_doc = max(k // len(document_groups), 3)  # At least 3 chunks per document
            
#             for doc_id, chunks in document_groups.items():
#                 # Sort chunks by relevance score within each document
#                 chunks.sort(key=lambda x: x.get("score", 0), reverse=True)
#                 # Take top chunks from each document
#                 balanced_results.extend(chunks[:chunks_per_doc])
#                 logger.info(f"Added {min(len(chunks), chunks_per_doc)} chunks from document {doc_id}")
            
#             # Sort all results by score and return top k
#             balanced_results.sort(key=lambda x: x.get("score", 0), reverse=True)
#             return balanced_results[:k]
#         else:
#             # Single document, return as-is
#             return documents[:k]
            
#     except Exception as e:
#         logger.error(f"Error in balanced document retrieval: {e}")
#         return documents[:k] if documents else []

# # Background task for periodic cleanup
# async def periodic_cleanup():
#     """Run cleanup every hour"""
#     while True:
#         await asyncio.sleep(3600)  # Sleep for 1 hour
#         try:
#             logger.info("Running periodic cleanup...")
#             # Clean up old sessions
#             rag_pipeline.cleanup_old_sessions(hours=24)
            
#             # Clean up inactive sessions
#             current_time = datetime.now()
#             sessions_to_remove = []
            
#             for session_id, info in active_sessions.items():
#                 if current_time - info["last_activity"] > timedelta(minutes=SESSION_TIMEOUT_MINUTES):
#                     sessions_to_remove.append(session_id)
            
#             for session_id in sessions_to_remove:
#                 try:
#                     rag_pipeline.cleanup_session(session_id)
#                     del active_sessions[session_id]
#                     logger.info(f"Cleaned up inactive session: {session_id}")
#                 except Exception as e:
#                     logger.error(f"Error cleaning up session {session_id}: {e}")
                    
#         except Exception as e:
#             logger.error(f"Periodic cleanup error: {e}")

# @app.on_event("startup")
# async def startup_event():
#     """Start background tasks on startup"""
#     asyncio.create_task(periodic_cleanup())
#     logger.info("Started periodic cleanup task")
#     logger.info("MCP Tools initialized: Sequential Thinking & Web Search")

# # Health check endpoint
# @app.get("/health")
# async def health_check():
#     """Check system health"""
#     return {
#         "status": "healthy",
#         "ollama_running": ollama_client.is_running(),
#         "active_sessions": len(active_sessions),
#         "max_sessions": MAX_CONCURRENT_USERS,
#         "components": {
#             "vector_store": "ready",
#             "embedder": "ready",
#             "reranker": "ready",
#             "mcp_tools": {
#                 "sequential_thinking": "ready",
#                 "web_search": "ready"
#             }
#         }
#     }

# # Session management
# @app.post("/sessions")
# async def create_session(title: str = "New Chat"):
#     """Create a new session with resource limits"""
#     # Check if we're at max capacity
#     if len(active_sessions) >= MAX_CONCURRENT_USERS:
#         # Try to clean up old sessions first
#         current_time = datetime.now()
#         cleaned = False
        
#         for session_id, info in list(active_sessions.items()):
#             if current_time - info["last_activity"] > timedelta(minutes=SESSION_TIMEOUT_MINUTES):
#                 try:
#                     await cleanup_session(session_id)
#                     cleaned = True
#                     break
#                 except:
#                     pass
        
#         if not cleaned and len(active_sessions) >= MAX_CONCURRENT_USERS:
#             raise HTTPException(
#                 status_code=503, 
#                 detail=f"Server at capacity. Maximum {MAX_CONCURRENT_USERS} concurrent users allowed."
#             )
    
#     # Create new session
#     session_id = rag_pipeline.session_manager.create_session(title)
    
#     # Track session
#     active_sessions[session_id] = {
#         "created_at": datetime.now(),
#         "last_activity": datetime.now(),
#         "title": title
#     }
    
#     return {
#         "session_id": session_id, 
#         "title": title,
#         "message": "Session created. Your documents will be isolated to this session only."
#     }

# @app.post("/sessions/{session_id}/cleanup")
# async def cleanup_session(session_id: str):
#     """Clean up all data for a session"""
#     try:
#         # Clean up the session
#         rag_pipeline.cleanup_session(session_id)
        
#         # Remove from active sessions
#         if session_id in active_sessions:
#             del active_sessions[session_id]
        
#         return {
#             "message": f"Session {session_id} cleaned up successfully",
#             "status": "success"
#         }
#     except Exception as e:
#         logger.error(f"Cleanup error for session {session_id}: {e}")
#         raise HTTPException(status_code=500, detail=str(e))

# @app.get("/sessions/{session_id}/stats")
# async def get_session_stats(session_id: str):
#     """Get statistics for a session"""
#     try:
#         import json
        
#         # Get metadata from sessions table
#         conn = sqlite3.connect(config.SESSION_DB_PATH)
#         cursor = conn.cursor()
        
#         cursor.execute('SELECT metadata FROM sessions WHERE id = ?', (session_id,))
#         result = cursor.fetchone()
        
#         if result and result[0]:
#             metadata = json.loads(result[0])
#             doc_count = metadata.get('document_count', 0)
#             chunk_count = metadata.get('chunk_count', 0)
            
#             conn.close()
            
#             logger.info(f"Session {session_id[:8]}... stats - Docs: {doc_count}, Chunks: {chunk_count}")
            
#             return {
#                 "session_id": session_id,
#                 "document_count": doc_count,
#                 "chunk_count": chunk_count
#             }
        
#         conn.close()
        
#         # No metadata found
#         return {
#             "session_id": session_id,
#             "document_count": 0,
#             "chunk_count": 0
#         }
        
#     except Exception as e:
#         logger.error(f"Error getting session stats: {e}")
#         return {
#             "session_id": session_id,
#             "document_count": 0,
#             "chunk_count": 0
#         }

# # Document upload and indexing
# @app.post("/upload/{session_id}")
# async def upload_documents(session_id: str, files: List[UploadFile] = File(...)):
#     """Upload documents for a specific session"""
#     # Update session activity
#     if session_id in active_sessions:
#         active_sessions[session_id]["last_activity"] = datetime.now()
    
#     # Create session-specific upload directory
#     upload_dir = Path("./shared_uploads") / session_id
#     upload_dir.mkdir(parents=True, exist_ok=True)
    
#     file_paths = []
#     for file in files:
#         # Save uploaded file
#         file_id = str(uuid.uuid4())[:8]
#         file_path = upload_dir / f"{file_id}_{file.filename}"
        
#         with open(file_path, "wb") as f:
#             shutil.copyfileobj(file.file, f)
        
#         file_paths.append(str(file_path))
#         logger.info(f"Uploaded file for session {session_id}: {file_path}")
    
#     return {
#         "message": f"Uploaded {len(files)} files",
#         "file_paths": file_paths,
#         "session_id": session_id
#     }

# @app.post("/index")
# async def index_documents(request: IndexRequest):
#     """Index uploaded documents for a specific session"""
#     try:
#         # Update session activity
#         if request.session_id in active_sessions:
#             active_sessions[request.session_id]["last_activity"] = datetime.now()
        
#         logger.info(f"Starting indexing for session {request.session_id}")
        
#         # Index with session isolation
#         result = rag_pipeline.index_documents(
#             request.file_paths,
#             session_id=request.session_id
#         )
        
#         # Only proceed if indexing was successful
#         if result.get("status") == "success":
#             # Create ONE index record for this batch
#             index_id = rag_pipeline.session_manager.create_index(
#                 name=f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
#                 document_count=result.get("documents_processed", 0)
#             )
            
#             # Link index to session
#             rag_pipeline.session_manager.link_index_to_session(request.session_id, index_id)
            
#             result["index_id"] = index_id
            
#             logger.info(f"✅ Indexing successful - Docs: {result['documents_processed']}, Chunks: {result['chunks_created']}")
#         else:
#             logger.error(f"❌ Indexing failed: {result.get('message', 'Unknown error')}")
#             raise HTTPException(status_code=500, detail=result.get("message", "Indexing failed"))
        
#         return result
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Indexing error: {e}")
#         import traceback
#         logger.error(traceback.format_exc())
#         raise HTTPException(status_code=500, detail=str(e))

# # # Enhanced query endpoint with MCP tools - FIXED VERSION with Multi-Document Support
# # @app.post("/query")
# # async def query_documents(request: QueryRequest):
# #     """Query indexed documents with MCP tools enhancement and multi-document support"""
# #     try:
# #         # Ensure session_id is provided
# #         if not request.session_id:
# #             raise HTTPException(
# #                 status_code=400,
# #                 detail="session_id is required. Please create a session first."
# #             )
        
# #         # Update session activity
# #         if request.session_id in active_sessions:
# #             active_sessions[request.session_id]["last_activity"] = datetime.now()

# #         # Extract tool flags from query
# #         clean_query, flag_sequential, flag_web = mcp_orchestrator.extract_tool_flags(request.query)

# #         # Detect news queries and enhance them
# #         if any(word in clean_query.lower() for word in ['news', 'latest', 'recent', 'update']):
# #             if 'latest' not in clean_query.lower() or 'news' not in clean_query.lower():
# #                 enhanced_query = f"{clean_query} 2025"
# #             else:
# #                 enhanced_query = clean_query
# #         else:
# #             enhanced_query = clean_query

# #         # IMPROVED: Smart query classification
# #         query_lower = clean_query.lower()

# #         # Define query types for better routing
# #         document_keywords = ['document', 'summarize', 'explain this', 'both documents', 'all documents', 'both files']
# #         external_keywords = ['weather', 'news', 'latest', 'current', 'today']

# #         is_document_query = any(keyword in query_lower for keyword in document_keywords)
# #         is_external_query = any(keyword in query_lower for keyword in external_keywords)
# #         is_summary_request = any(word in query_lower for word in ['summarize', 'summary', 'both', 'all documents'])

# #         # Determine which tools to use with smart logic
# #         use_sequential = flag_sequential or request.use_sequential_thinking
# #         use_web = flag_web or request.use_web_search

# #         # FIXED: Smart web search logic - only use web for external queries or when no documents
# #         should_use_web_search = use_web and (is_external_query or is_document_query == False)

# #         # Get session context (recent messages) - DEFINE THIS HERE
# #         recent_messages = rag_pipeline.session_manager.get_session_messages(
# #             request.session_id, limit=5
# #         )
# #         context = "\n".join([msg["content"] for msg in recent_messages[-3:] if msg["role"] == "assistant"])

# #         # Apply MCP tools if enabled
# #         enhanced_result = None
# #         if use_sequential or should_use_web_search:
# #             # Enhance query with MCP tools
# #             enhanced_result = enhance_query_with_mcp_tools(
# #                 enhanced_query,
# #                 session_context=context,
# #                 enable_sequential=use_sequential,
# #                 enable_web_search=should_use_web_search
# #             )
            
# #             logger.info(f"Applied MCP tools: {enhanced_result['tools_used']}")

# #         # FIXED: Smart document retrieval with balanced multi-document support
# #         documents = []
# #         if not is_external_query:
# #             # Use the new balanced retrieval function
# #             documents = get_balanced_documents(
# #                 rag_pipeline,
# #                 query=clean_query,
# #                 session_id=request.session_id,
# #                 k=request.retrieval_k,
# #                 use_reranker=request.use_reranker
# #             )
        
# #         # Handle different query scenarios
# #         if is_external_query:
# #             # EXTERNAL QUERY: Prioritize web search, ignore documents
# #             if not enhanced_result or "web_search" not in enhanced_result.get("tool_outputs", {}):
# #                 return {
# #                     "answer": "This appears to be a request for current external information, but web search is not enabled. Please enable web search to get current information.",
# #                     "source_documents": [],
# #                     "session_id": request.session_id,
# #                     "mcp_tools_used": []
# #                 }
            
# #             web_data = enhanced_result["tool_outputs"]["web_search"]
# #             web_context = web_data.get("context", "No web results available")
            
# #             # Build web-only prompt
# #             prompt_sections = []
            
# #             # Add Sequential Thinking if available
# #             if enhanced_result and "sequential_thinking" in enhanced_result.get("tool_outputs", {}):
# #                 thinking = enhanced_result["tool_outputs"]["sequential_thinking"]
# #                 prompt_sections.append("[Sequential Thinking Analysis]")
# #                 if "approach" in thinking:
# #                     prompt_sections.append(f"Approach: {thinking['approach']}")
# #                 if "thinking_steps" in thinking:
# #                     prompt_sections.append("\nThinking Steps:")
# #                     for step in thinking["thinking_steps"]:
# #                         prompt_sections.append(f"- {step.get('step', 'Unknown')}: {step.get('result', 'No result')}")
# #                 prompt_sections.append("")
            
# #             # Add web search results
# #             prompt_sections.append("[Current Web Information]")
# #             if web_data.get("results"):
# #                 for i, result in enumerate(web_data["results"][:5], 1):
# #                     prompt_sections.append(f"{i}. {result.get('title', 'No title')}")
# #                     prompt_sections.append(f"   {result.get('snippet', 'No snippet')[:300]}")
# #                     prompt_sections.append(f"   Source: {result.get('displayLink', 'Unknown')}")
# #                     prompt_sections.append("")
            
# #             final_prompt = f"""{chr(10).join(prompt_sections)}

# # User Question: {clean_query}

# # Instructions:
# # - Use ONLY the current web information provided above
# # - Do NOT reference any uploaded documents or internal knowledge
# # - Provide accurate, up-to-date information based on web sources
# # - If web information is insufficient, clearly state what's missing
# # - Cite sources when providing specific facts

# # Answer:"""
            
# #         elif is_document_query and documents:
# #             # DOCUMENT QUERY: Use documents with improved multi-document context building
            
# #             # IMPROVED: Build context ensuring all documents are represented
# #             doc_groups = {}
# #             for doc in documents:
# #                 doc_id = doc.get("document_id", "unknown")
# #                 if doc_id not in doc_groups:
# #                     doc_groups[doc_id] = []
# #                 doc_groups[doc_id].append(doc["text"])
            
# #             # Build context parts ensuring representation from all documents
# #             context_parts = []
# #             for doc_id, chunks in doc_groups.items():
# #                 context_parts.append(f"[Document ID: {doc_id}]")
# #                 # Take top chunks from each document
# #                 context_parts.extend(chunks[:8])  # More chunks per document for better coverage
# #                 context_parts.append("")
            
# #             doc_context = "\n\n".join(context_parts)
            
# #             prompt_sections = []
            
# #             # Add Sequential Thinking if available
# #             if enhanced_result and "sequential_thinking" in enhanced_result.get("tool_outputs", {}):
# #                 thinking = enhanced_result["tool_outputs"]["sequential_thinking"]
# #                 prompt_sections.append("[Sequential Thinking Analysis]")
# #                 if "approach" in thinking:
# #                     prompt_sections.append(f"Approach: {thinking['approach']}")
# #                 if "thinking_steps" in thinking:
# #                     prompt_sections.append("\nThinking Steps:")
# #                     for step in thinking["thinking_steps"]:
# #                         prompt_sections.append(f"- {step.get('step', 'Unknown')}: {step.get('result', 'No result')}")
# #                 prompt_sections.append("")
            
# #             thinking_context = "\n".join(prompt_sections) if prompt_sections else ""
            
# #             # ENHANCED: Multi-document summarization prompt
# #             if is_summary_request:
# #                 final_prompt = f"""{thinking_context}

# # Document Context:
# # {doc_context}

# # User Question: {clean_query}

# # Instructions:
# # - Provide a comprehensive summary that covers ALL documents provided above
# # - Ensure you include key points from EACH document (note the Document ID labels)
# # - Create a balanced summary that represents all uploaded documents
# # - If you notice content from only one document, explicitly mention this limitation
# # - Structure your summary to clearly show insights from multiple sources
# # - Reference document IDs when highlighting specific points

# # Answer:"""
# #             else:
# #                 final_prompt = f"""{thinking_context}

# # Document Context:
# # {doc_context}

# # User Question: {clean_query}

# # Instructions:
# # - Provide a comprehensive answer based on ALL document content above
# # - Be specific and reference key points from the documents
# # - When relevant, mention which document (by ID) contains specific information
# # - Follow any analysis framework provided above
# # - If information is insufficient, clearly state what's missing

# # Answer:"""
            
# #         else:
# #             # GENERAL/MIXED QUERY: Handle lack of documents or hybrid needs
# #             if not documents and not enhanced_result:
# #                 return {
# #                     "answer": "No documents found in this session. Please upload and index documents first, or enable web search for external information.",
# #                     "source_documents": [],
# #                     "session_id": request.session_id,
# #                     "mcp_tools_used": []
# #                 }
            
# #             # Build context from available sources
# #             context_parts = []
            
# #             # Add Sequential Thinking if available
# #             if enhanced_result and "sequential_thinking" in enhanced_result.get("tool_outputs", {}):
# #                 thinking = enhanced_result["tool_outputs"]["sequential_thinking"]
# #                 context_parts.append("[Sequential Thinking Analysis]")
# #                 if "approach" in thinking:
# #                     context_parts.append(f"Approach: {thinking['approach']}")
# #                 if "thinking_steps" in thinking:
# #                     context_parts.append("\nThinking Steps:")
# #                     for step in thinking["thinking_steps"]:
# #                         context_parts.append(f"- {step.get('step', 'Unknown')}: {step.get('result', 'No result')}")
# #                 context_parts.append("")
            
# #             # Add document context if available with multi-document support
# #             if documents:
# #                 doc_groups = {}
# #                 for doc in documents:
# #                     doc_id = doc.get("document_id", "unknown")
# #                     if doc_id not in doc_groups:
# #                         doc_groups[doc_id] = []
# #                     doc_groups[doc_id].append(doc["text"])
                
# #                 context_parts.append("[Document Context]")
# #                 for doc_id, chunks in doc_groups.items():
# #                     context_parts.append(f"Document {doc_id}:")
# #                     context_parts.extend(chunks[:5])  # Top 5 chunks per document
# #                 context_parts.append("")
            
# #             # Add web context if available
# #             if enhanced_result and "web_search" in enhanced_result.get("tool_outputs", {}):
# #                 web_data = enhanced_result["tool_outputs"]["web_search"]
# #                 if web_data.get("results"):
# #                     context_parts.append("[Current Web Information]")
# #                     for i, result in enumerate(web_data["results"][:3], 1):
# #                         context_parts.append(f"{i}. {result.get('title', 'No title')}")
# #                         context_parts.append(f"   {result.get('snippet', 'No snippet')[:200]}")
# #                         context_parts.append(f"   Source: {result.get('displayLink', 'Unknown')}")
# #                     context_parts.append("")
            
# #             combined_context = "\n".join(context_parts) if context_parts else "No specific context available."
            
# #             final_prompt = f"""{combined_context}

# # User Question: {clean_query}

# # Instructions:
# # - Use the most relevant context above to answer the question
# # - If the question requires current information, prioritize web sources
# # - If the question is about uploaded documents, prioritize document sources
# # - Be specific and cite your sources (including document IDs when relevant)
# # - If information is insufficient, clearly state what's missing

# # Answer:"""
        
# #         logger.info(f"Sending prompt to Ollama (length: {len(final_prompt)} chars)")
# #         logger.debug(f"Query type - Document: {is_document_query}, External: {is_external_query}, Summary: {is_summary_request}")
        
# #         # Generate response with explicit model specification
# #         answer = ollama_client.generate(final_prompt, model="qwen2:7b")
        
# #         # Check if we got a valid response
# #         if not answer or answer.strip() == "" or answer.startswith("Error"):
# #             logger.error(f"No valid response from Ollama: {answer}")
# #             answer = "I apologize, but I couldn't generate a proper response. Please try again."
# #         else:
# #             logger.info(f"Successfully generated response: {len(answer)} chars")
        
# #         # Format MCP tools indicator for response
# #         tools_used_text = ""
# #         if enhanced_result and enhanced_result.get("tools_used"):
# #             tools_indicators = []
# #             if "Sequential Thinking" in enhanced_result["tools_used"]:
# #                 tools_indicators.append("🧠 Sequential Thinking")
# #             if "Web Search" in enhanced_result["tools_used"]:
# #                 tools_indicators.append("🔍 Web Search")
            
# #             if tools_indicators:
# #                 tools_used_text = f"[{' • '.join(tools_indicators)}]\n\n"
        
# #         final_answer = answer
        
# #         # Save to session
# #         rag_pipeline.session_manager.add_message(request.session_id, "user", clean_query)
# #         rag_pipeline.session_manager.add_message(request.session_id, "assistant", final_answer)
        
# #         # Prepare response
# #         response_data = {
# #             "answer": final_answer,
# #             "source_documents": documents[:10] if documents else [],  # Return more source docs for transparency
# #             "session_id": request.session_id,
# #             "mcp_tools_used": enhanced_result.get("tools_used", []) if enhanced_result else [],
# #             "documents_used": len(set(doc.get("document_id") for doc in documents)) if documents else 0  # NEW: Track unique documents used
# #         }
        
# #         # Add web search results if available
# #         if enhanced_result and "web_search" in enhanced_result.get("tool_outputs", {}):
# #             response_data["web_sources"] = enhanced_result["tool_outputs"]["web_search"].get("results", [])
        
# #         return response_data
        
# #     except HTTPException:
# #         raise
# #     except Exception as e:
# #         logger.error(f"Query error: {e}")
# #         import traceback
# #         logger.error(f"Traceback: {traceback.format_exc()}")
# #         raise HTTPException(status_code=500, detail=str(e))

# # Enhanced query endpoint with MCP tools - FIXED VERSION with Memory Support
# @app.post("/query")
# async def query_documents(request: QueryRequest):
#     """Query indexed documents with MCP tools enhancement and proper conversation memory"""
#     try:
#         # Ensure session_id is provided
#         if not request.session_id:
#             raise HTTPException(
#                 status_code=400,
#                 detail="session_id is required. Please create a session first."
#             )
        
#         # Update session activity
#         if request.session_id in active_sessions:
#             active_sessions[request.session_id]["last_activity"] = datetime.now()

#         # Extract tool flags from query
#         clean_query, flag_sequential, flag_web = mcp_orchestrator.extract_tool_flags(request.query)

#         # FIXED: Get FULL conversation history for context
#         recent_messages = rag_pipeline.session_manager.get_session_messages(
#             request.session_id, limit=10  # Get more messages
#         )
        
#         # Format conversation history properly
#         conversation_history = []
#         for msg in recent_messages[-8:]:  # Last 8 messages (4 exchanges)
#             role = "User" if msg["role"] == "user" else "Assistant"
#             conversation_history.append(f"{role}: {msg['content']}")
        
#         conversation_context = "\n".join(conversation_history) if conversation_history else "No previous conversation."

#         # Detect news queries and enhance them
#         if any(word in clean_query.lower() for word in ['news', 'latest', 'recent', 'update']):
#             if 'latest' not in clean_query.lower() or 'news' not in clean_query.lower():
#                 enhanced_query = f"{clean_query} 2025"
#             else:
#                 enhanced_query = clean_query
#         else:
#             enhanced_query = clean_query

#         # IMPROVED: Smart query classification
#         query_lower = clean_query.lower()

#         # Define query types for better routing
#         document_keywords = ['document', 'summarize', 'explain this', 'both documents', 'all documents', 'both files']
#         external_keywords = ['weather', 'news', 'latest', 'current', 'today']
#         memory_keywords = ['first', 'earlier', 'before', 'previous', 'remember', 'recall', 'memory', 'conversation', 'chat', 'said']

#         is_document_query = any(keyword in query_lower for keyword in document_keywords)
#         is_external_query = any(keyword in query_lower for keyword in external_keywords)
#         is_memory_query = any(keyword in query_lower for keyword in memory_keywords)
#         is_summary_request = any(word in query_lower for word in ['summarize', 'summary', 'both', 'all documents'])

#         # Determine which tools to use with smart logic
#         use_sequential = flag_sequential or request.use_sequential_thinking
#         use_web = flag_web or request.use_web_search

#         # FIXED: Smart web search logic - only use web for external queries or when no documents
#         should_use_web_search = use_web and (is_external_query or is_document_query == False) and not is_memory_query

#         # Apply MCP tools if enabled
#         enhanced_result = None
#         if use_sequential or should_use_web_search:
#             # Enhance query with MCP tools
#             enhanced_result = enhance_query_with_mcp_tools(
#                 enhanced_query,
#                 session_context=conversation_context,  # Pass full conversation context
#                 enable_sequential=use_sequential,
#                 enable_web_search=should_use_web_search
#             )
            
#             logger.info(f"Applied MCP tools: {enhanced_result['tools_used']}")

#         # FIXED: Handle memory queries first (BEFORE document retrieval)
#         if is_memory_query:
#             # MEMORY QUERY: Use conversation history directly
#             prompt_sections = []
            
#             # Add Sequential Thinking if available
#             if enhanced_result and "sequential_thinking" in enhanced_result.get("tool_outputs", {}):
#                 thinking = enhanced_result["tool_outputs"]["sequential_thinking"]
#                 prompt_sections.append("[Sequential Thinking Analysis]")
#                 if "approach" in thinking:
#                     prompt_sections.append(f"Approach: {thinking['approach']}")
#                 if "thinking_steps" in thinking:
#                     prompt_sections.append("\nThinking Steps:")
#                     for step in thinking["thinking_steps"]:
#                         prompt_sections.append(f"- {step.get('step', 'Unknown')}: {step.get('result', 'No result')}")
#                 prompt_sections.append("")
            
#             # Add conversation history
#             prompt_sections.append("[Previous Conversation History]")
#             if conversation_history:
#                 prompt_sections.extend(conversation_history)
#             else:
#                 prompt_sections.append("No previous conversation found.")
#             prompt_sections.append("")
            
#             final_prompt = f"""{chr(10).join(prompt_sections)}

# User Question: {clean_query}

# Instructions:
# - Use the conversation history above to answer questions about previous interactions
# - Be specific about what was said and when in the conversation
# - If asking about the "first" question, refer to the earliest user message
# - Reference specific messages from the conversation history
# - If you cannot find the information in the conversation history, say so clearly

# Answer:"""

#         else:
#             # FIXED: Smart document retrieval with balanced multi-document support
#             documents = []
#             if not is_external_query:
#                 # Use the balanced retrieval function
#                 documents = get_balanced_documents(
#                     rag_pipeline,
#                     query=clean_query,
#                     session_id=request.session_id,
#                     k=request.retrieval_k,
#                     use_reranker=request.use_reranker
#                 )
            
#             # Handle different query scenarios
#             if is_external_query:
#                 # EXTERNAL QUERY: Prioritize web search, ignore documents
#                 if not enhanced_result or "web_search" not in enhanced_result.get("tool_outputs", {}):
#                     return {
#                         "answer": "This appears to be a request for current external information, but web search is not enabled. Please enable web search to get current information.",
#                         "source_documents": [],
#                         "session_id": request.session_id,
#                         "mcp_tools_used": []
#                     }
                
#                 web_data = enhanced_result["tool_outputs"]["web_search"]
#                 web_context = web_data.get("context", "No web results available")
                
#                 # Build web-only prompt
#                 prompt_sections = []
                
#                 # Add Sequential Thinking if available
#                 if enhanced_result and "sequential_thinking" in enhanced_result.get("tool_outputs", {}):
#                     thinking = enhanced_result["tool_outputs"]["sequential_thinking"]
#                     prompt_sections.append("[Sequential Thinking Analysis]")
#                     if "approach" in thinking:
#                         prompt_sections.append(f"Approach: {thinking['approach']}")
#                     if "thinking_steps" in thinking:
#                         prompt_sections.append("\nThinking Steps:")
#                         for step in thinking["thinking_steps"]:
#                             prompt_sections.append(f"- {step.get('step', 'Unknown')}: {step.get('result', 'No result')}")
#                     prompt_sections.append("")
                
#                 # Add web search results
#                 prompt_sections.append("[Current Web Information]")
#                 if web_data.get("results"):
#                     for i, result in enumerate(web_data["results"][:5], 1):
#                         prompt_sections.append(f"{i}. {result.get('title', 'No title')}")
#                         prompt_sections.append(f"   {result.get('snippet', 'No snippet')[:300]}")
#                         prompt_sections.append(f"   Source: {result.get('displayLink', 'Unknown')}")
#                         prompt_sections.append("")
                
#                 final_prompt = f"""{chr(10).join(prompt_sections)}

# User Question: {clean_query}

# Instructions:
# - Use ONLY the current web information provided above
# - Do NOT reference any uploaded documents or internal knowledge
# - Provide accurate, up-to-date information based on web sources
# - If web information is insufficient, clearly state what's missing
# - Cite sources when providing specific facts

# Answer:"""
                
#             elif is_document_query and documents:
#                 # DOCUMENT QUERY: Use documents with improved multi-document context building
                
#                 # IMPROVED: Build context ensuring all documents are represented
#                 doc_groups = {}
#                 for doc in documents:
#                     doc_id = doc.get("document_id", "unknown")
#                     if doc_id not in doc_groups:
#                         doc_groups[doc_id] = []
#                     doc_groups[doc_id].append(doc["text"])
                
#                 # Build context parts ensuring representation from all documents
#                 context_parts = []
#                 for doc_id, chunks in doc_groups.items():
#                     context_parts.append(f"[Document ID: {doc_id}]")
#                     # Take top chunks from each document
#                     context_parts.extend(chunks[:8])  # More chunks per document for better coverage
#                     context_parts.append("")
                
#                 doc_context = "\n\n".join(context_parts)
                
#                 prompt_sections = []
                
#                 # Add Sequential Thinking if available
#                 if enhanced_result and "sequential_thinking" in enhanced_result.get("tool_outputs", {}):
#                     thinking = enhanced_result["tool_outputs"]["sequential_thinking"]
#                     prompt_sections.append("[Sequential Thinking Analysis]")
#                     if "approach" in thinking:
#                         prompt_sections.append(f"Approach: {thinking['approach']}")
#                     if "thinking_steps" in thinking:
#                         prompt_sections.append("\nThinking Steps:")
#                         for step in thinking["thinking_steps"]:
#                             prompt_sections.append(f"- {step.get('step', 'Unknown')}: {step.get('result', 'No result')}")
#                     prompt_sections.append("")
                
#                 thinking_context = "\n".join(prompt_sections) if prompt_sections else ""
                
#                 # ENHANCED: Multi-document summarization prompt
#                 if is_summary_request:
#                     final_prompt = f"""{thinking_context}

# Document Context:
# {doc_context}

# User Question: {clean_query}

# Instructions:
# - Provide a comprehensive summary that covers ALL documents provided above
# - Ensure you include key points from EACH document (note the Document ID labels)
# - Create a balanced summary that represents all uploaded documents
# - If you notice content from only one document, explicitly mention this limitation
# - Structure your summary to clearly show insights from multiple sources
# - Reference document IDs when highlighting specific points

# Answer:"""
#                 else:
#                     final_prompt = f"""{thinking_context}

# Document Context:
# {doc_context}

# User Question: {clean_query}

# Instructions:
# - Provide a comprehensive answer based on ALL document content above
# - Be specific and reference key points from the documents
# - When relevant, mention which document (by ID) contains specific information
# - Follow any analysis framework provided above
# - If information is insufficient, clearly state what's missing

# Answer:"""
                
#             else:
#                 # GENERAL/MIXED QUERY: Handle lack of documents or hybrid needs
#                 if not documents and not enhanced_result:
#                     return {
#                         "answer": "No documents found in this session. Please upload and index documents first, or enable web search for external information.",
#                         "source_documents": [],
#                         "session_id": request.session_id,
#                         "mcp_tools_used": []
#                     }
                
#                 # Build context from available sources
#                 context_parts = []
                
#                 # ADDED: Include conversation context for general queries
#                 if conversation_history and not is_external_query:
#                     context_parts.append("[Previous Conversation]")
#                     context_parts.extend(conversation_history[-4:])  # Last 4 messages
#                     context_parts.append("")
                
#                 # Add Sequential Thinking if available
#                 if enhanced_result and "sequential_thinking" in enhanced_result.get("tool_outputs", {}):
#                     thinking = enhanced_result["tool_outputs"]["sequential_thinking"]
#                     context_parts.append("[Sequential Thinking Analysis]")
#                     if "approach" in thinking:
#                         context_parts.append(f"Approach: {thinking['approach']}")
#                     if "thinking_steps" in thinking:
#                         context_parts.append("\nThinking Steps:")
#                         for step in thinking["thinking_steps"]:
#                             context_parts.append(f"- {step.get('step', 'Unknown')}: {step.get('result', 'No result')}")
#                     context_parts.append("")
                
#                 # Add document context if available with multi-document support
#                 if documents:
#                     doc_groups = {}
#                     for doc in documents:
#                         doc_id = doc.get("document_id", "unknown")
#                         if doc_id not in doc_groups:
#                             doc_groups[doc_id] = []
#                         doc_groups[doc_id].append(doc["text"])
                    
#                     context_parts.append("[Document Context]")
#                     for doc_id, chunks in doc_groups.items():
#                         context_parts.append(f"Document {doc_id}:")
#                         context_parts.extend(chunks[:5])  # Top 5 chunks per document
#                     context_parts.append("")
                
#                 # Add web context if available
#                 if enhanced_result and "web_search" in enhanced_result.get("tool_outputs", {}):
#                     web_data = enhanced_result["tool_outputs"]["web_search"]
#                     if web_data.get("results"):
#                         context_parts.append("[Current Web Information]")
#                         for i, result in enumerate(web_data["results"][:3], 1):
#                             context_parts.append(f"{i}. {result.get('title', 'No title')}")
#                             context_parts.append(f"   {result.get('snippet', 'No snippet')[:200]}")
#                             context_parts.append(f"   Source: {result.get('displayLink', 'Unknown')}")
#                         context_parts.append("")
                
#                 combined_context = "\n".join(context_parts) if context_parts else "No specific context available."
                
#                 final_prompt = f"""{combined_context}

# User Question: {clean_query}

# Instructions:
# - Use the most relevant context above to answer the question
# - If the question requires current information, prioritize web sources
# - If the question is about uploaded documents, prioritize document sources
# - If the question is about previous conversation, use the conversation history
# - Be specific and cite your sources (including document IDs when relevant)
# - If information is insufficient, clearly state what's missing

# Answer:"""
        
#         logger.info(f"Sending prompt to Ollama (length: {len(final_prompt)} chars)")
#         logger.debug(f"Query type - Document: {is_document_query}, External: {is_external_query}, Memory: {is_memory_query}, Summary: {is_summary_request}")
        
#         # Generate response with explicit model specification
#         answer = ollama_client.generate(final_prompt, model="qwen2:7b")
        
#         # Check if we got a valid response
#         if not answer or answer.strip() == "" or answer.startswith("Error"):
#             logger.error(f"No valid response from Ollama: {answer}")
#             answer = "I apologize, but I couldn't generate a proper response. Please try again."
#         else:
#             logger.info(f"Successfully generated response: {len(answer)} chars")
        
#         # Format MCP tools indicator for response
#         tools_used_text = ""
#         if enhanced_result and enhanced_result.get("tools_used"):
#             tools_indicators = []
#             if "Sequential Thinking" in enhanced_result["tools_used"]:
#                 tools_indicators.append("🧠 Sequential Thinking")
#             if "Web Search" in enhanced_result["tools_used"]:
#                 tools_indicators.append("🔍 Web Search")
            
#             if tools_indicators:
#                 tools_used_text = f"[{' • '.join(tools_indicators)}]\n\n"
        
#         final_answer = answer
        
#         # Save to session
#         rag_pipeline.session_manager.add_message(request.session_id, "user", clean_query)
#         rag_pipeline.session_manager.add_message(request.session_id, "assistant", final_answer)
        
#         # Prepare response
#         response_data = {
#             "answer": final_answer,
#             "source_documents": documents[:10] if documents and not is_memory_query else [],  # Don't show docs for memory queries
#             "session_id": request.session_id,
#             "mcp_tools_used": enhanced_result.get("tools_used", []) if enhanced_result else [],
#             "documents_used": len(set(doc.get("document_id") for doc in documents)) if documents and not is_memory_query else 0,
#             "conversation_context_used": len(conversation_history) > 0  # NEW: Indicate if conversation context was used
#         }
        
#         # Add web search results if available
#         if enhanced_result and "web_search" in enhanced_result.get("tool_outputs", {}):
#             response_data["web_sources"] = enhanced_result["tool_outputs"]["web_search"].get("results", [])
        
#         return response_data
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Query error: {e}")
#         import traceback
#         logger.error(f"Traceback: {traceback.format_exc()}")
#         raise HTTPException(status_code=500, detail=str(e))

# @app.post("/chat")
# async def chat(request: ChatRequest):
#     """Chat endpoint with MCP tools support"""
#     try:
#         # Ensure session_id
#         if not request.session_id:
#             session_response = await create_session()
#             request.session_id = session_response["session_id"]
        
#         # Update session activity
#         if request.session_id in active_sessions:
#             active_sessions[request.session_id]["last_activity"] = datetime.now()
        
#         # Get the latest message
#         latest_message = request.messages[-1] if request.messages else None
#         if not latest_message:
#             raise HTTPException(status_code=400, detail="No messages provided")
        
#         # Apply MCP tools if enabled
#         enhanced_content = latest_message.content
#         tools_info = ""
        
#         if request.use_sequential_thinking or request.use_web_search:
#             enhanced_result = enhance_query_with_mcp_tools(
#                 latest_message.content,
#                 enable_sequential=request.use_sequential_thinking,
#                 enable_web_search=request.use_web_search
#             )
            
#             if enhanced_result["tools_used"]:
#                 tools_indicators = []
#                 if "Sequential Thinking" in enhanced_result["tools_used"]:
#                     tools_indicators.append("🧠 Sequential Thinking")
#                 if "Web Search" in enhanced_result["tools_used"]:
#                     tools_indicators.append("🔍 Web Search")
#                 tools_info = f"[{' • '.join(tools_indicators)}]\n\n"
            
#             # Add tool context to the message
#             if "sequential_thinking" in enhanced_result["tool_outputs"]:
#                 thinking = enhanced_result["tool_outputs"]["sequential_thinking"]
#                 enhanced_content += f"\n\n[Approach: {thinking['approach']}]"
            
#             if "web_search" in enhanced_result["tool_outputs"]:
#                 web_context = enhanced_result["tool_outputs"]["web_search"]["context"]
#                 enhanced_content += f"\n\n{web_context}"
        
#         # Convert messages to Ollama format with enhancement
#         messages = []
        
#         # Add system message
#         messages.append({
#             "role": "system",
#             "content": "You are a helpful assistant with access to documents and advanced reasoning capabilities. You can only access documents uploaded in this specific session."
#         })
        
#         # Add conversation history
#         for msg in request.messages[:-1]:
#             messages.append({"role": msg.role, "content": msg.content})
        
#         # Add the enhanced latest message
#         messages.append({"role": latest_message.role, "content": enhanced_content})
        
#         # Get response from Ollama
#         response = ollama_client.chat(messages, model=request.model or config.GENERATION_MODEL)
        
#         # Add tools indicator to response
#         final_response = tools_info + response
        
#         # Save to session
#         rag_pipeline.session_manager.add_message(
#             request.session_id, 
#             latest_message.role, 
#             latest_message.content
#         )
#         rag_pipeline.session_manager.add_message(
#             request.session_id, 
#             "assistant", 
#             final_response
#         )
        
#         return {
#             "response": final_response, 
#             "session_id": request.session_id,
#             "mcp_tools_used": enhanced_result["tools_used"] if request.use_sequential_thinking or request.use_web_search else []
#         }
        
#     except Exception as e:
#         logger.error(f"Chat error: {e}")

"""
Enhanced FastAPI Server with MCP Tools Integration
Path: rag_system/api/server.py
"""
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import json
import sqlite3
import logging
import uvicorn
from pathlib import Path
import shutil
import uuid
import asyncio
import re
from datetime import datetime, timedelta

from rag_system.core.rag_pipeline import RAGPipeline
from rag_system.utils.ollama_client import OllamaClient
from rag_system.utils.mcp_tools import MCPToolsOrchestrator, enhance_query_with_mcp_tools
from rag_system.config.settings import config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Enterprise RAG API with MCP Tools", version="3.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
rag_pipeline = RAGPipeline()
ollama_client = OllamaClient(host=config.OLLAMA_HOST)
mcp_orchestrator = MCPToolsOrchestrator()

# Session tracking
active_sessions = {}
MAX_CONCURRENT_USERS = 5
SESSION_TIMEOUT_MINUTES = 30

# Request/Response models
class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    retrieval_k: Optional[int] = 50  # INCREASED from 20
    use_reranker: Optional[bool] = True
    use_sequential_thinking: Optional[bool] = False
    use_web_search: Optional[bool] = False

class IndexRequest(BaseModel):
    file_paths: List[str]
    session_id: str
    index_name: Optional[str] = "default"

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    session_id: Optional[str] = None
    model: Optional[str] = "qwen3:8b"
    use_sequential_thinking: Optional[bool] = False
    use_web_search: Optional[bool] = False

class SessionResponse(BaseModel):
    session_id: str
    title: str
    created_at: str
    updated_at: str

# ADDED: Function to ensure diverse document retrieval
def get_balanced_documents(rag_pipeline, query: str, session_id: str, k: int = 50, use_reranker: bool = True) -> List[Dict[str, Any]]:
    """Retrieve documents ensuring balanced representation from all uploaded documents"""
    try:
        # Get more documents initially
        initial_k = min(k * 2, 100)
        documents = rag_pipeline.retrieve(
            query=query,
            session_id=session_id,
            k=initial_k,
            use_reranker=use_reranker
        )
        
        if not documents:
            return []
        
        # Group documents by document_id to ensure diversity
        document_groups = {}
        for doc in documents:
            doc_id = doc.get("document_id", "unknown")
            if doc_id not in document_groups:
                document_groups[doc_id] = []
            document_groups[doc_id].append(doc)
        
        logger.info(f"Retrieved chunks from {len(document_groups)} unique documents")
        
        # If we have multiple documents, ensure balanced representation
        if len(document_groups) > 1:
            balanced_results = []
            chunks_per_doc = max(k // len(document_groups), 3)  # At least 3 chunks per document
            
            for doc_id, chunks in document_groups.items():
                # Sort chunks by relevance score within each document
                chunks.sort(key=lambda x: x.get("score", 0), reverse=True)
                # Take top chunks from each document
                balanced_results.extend(chunks[:chunks_per_doc])
                logger.info(f"Added {min(len(chunks), chunks_per_doc)} chunks from document {doc_id}")
            
            # Sort all results by score and return top k
            balanced_results.sort(key=lambda x: x.get("score", 0), reverse=True)
            return balanced_results[:k]
        else:
            # Single document, return as-is
            return documents[:k]
            
    except Exception as e:
        logger.error(f"Error in balanced document retrieval: {e}")
        return documents[:k] if documents else []

# Background task for periodic cleanup
async def periodic_cleanup():
    """Run cleanup every hour"""
    while True:
        await asyncio.sleep(3600)  # Sleep for 1 hour
        try:
            logger.info("Running periodic cleanup...")
            # Clean up old sessions
            rag_pipeline.cleanup_old_sessions(hours=24)
            
            # Clean up inactive sessions
            current_time = datetime.now()
            sessions_to_remove = []
            
            for session_id, info in active_sessions.items():
                if current_time - info["last_activity"] > timedelta(minutes=SESSION_TIMEOUT_MINUTES):
                    sessions_to_remove.append(session_id)
            
            for session_id in sessions_to_remove:
                try:
                    rag_pipeline.cleanup_session(session_id)
                    del active_sessions[session_id]
                    logger.info(f"Cleaned up inactive session: {session_id}")
                except Exception as e:
                    logger.error(f"Error cleaning up session {session_id}: {e}")
                    
        except Exception as e:
            logger.error(f"Periodic cleanup error: {e}")

@app.on_event("startup")
async def startup_event():
    """Start background tasks on startup"""
    asyncio.create_task(periodic_cleanup())
    logger.info("Started periodic cleanup task")
    logger.info("MCP Tools initialized: Sequential Thinking & Web Search")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Check system health"""
    return {
        "status": "healthy",
        "ollama_running": ollama_client.is_running(),
        "active_sessions": len(active_sessions),
        "max_sessions": MAX_CONCURRENT_USERS,
        "components": {
            "vector_store": "ready",
            "embedder": "ready",
            "reranker": "ready",
            "mcp_tools": {
                "sequential_thinking": "ready",
                "web_search": "ready"
            }
        }
    }

# Session management
@app.post("/sessions")
async def create_session(title: str = "New Chat"):
    """Create a new session with resource limits"""
    # Check if we're at max capacity
    if len(active_sessions) >= MAX_CONCURRENT_USERS:
        # Try to clean up old sessions first
        current_time = datetime.now()
        cleaned = False
        
        for session_id, info in list(active_sessions.items()):
            if current_time - info["last_activity"] > timedelta(minutes=SESSION_TIMEOUT_MINUTES):
                try:
                    await cleanup_session(session_id)
                    cleaned = True
                    break
                except:
                    pass
        
        if not cleaned and len(active_sessions) >= MAX_CONCURRENT_USERS:
            raise HTTPException(
                status_code=503, 
                detail=f"Server at capacity. Maximum {MAX_CONCURRENT_USERS} concurrent users allowed."
            )
    
    # Create new session
    session_id = rag_pipeline.session_manager.create_session(title)
    
    # Track session
    active_sessions[session_id] = {
        "created_at": datetime.now(),
        "last_activity": datetime.now(),
        "title": title
    }
    
    return {
        "session_id": session_id, 
        "title": title,
        "message": "Session created. Your documents will be isolated to this session only."
    }

@app.post("/sessions/{session_id}/cleanup")
async def cleanup_session(session_id: str):
    """Clean up all data for a session"""
    try:
        # Clean up the session
        rag_pipeline.cleanup_session(session_id)
        
        # Remove from active sessions
        if session_id in active_sessions:
            del active_sessions[session_id]
        
        return {
            "message": f"Session {session_id} cleaned up successfully",
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Cleanup error for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions/{session_id}/stats")
async def get_session_stats(session_id: str):
    """Get statistics for a session"""
    try:
        import json
        
        # Get metadata from sessions table
        conn = sqlite3.connect(config.SESSION_DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('SELECT metadata FROM sessions WHERE id = ?', (session_id,))
        result = cursor.fetchone()
        
        if result and result[0]:
            metadata = json.loads(result[0])
            doc_count = metadata.get('document_count', 0)
            chunk_count = metadata.get('chunk_count', 0)
            
            conn.close()
            
            logger.info(f"Session {session_id[:8]}... stats - Docs: {doc_count}, Chunks: {chunk_count}")
            
            return {
                "session_id": session_id,
                "document_count": doc_count,
                "chunk_count": chunk_count
            }
        
        conn.close()
        
        # No metadata found
        return {
            "session_id": session_id,
            "document_count": 0,
            "chunk_count": 0
        }
        
    except Exception as e:
        logger.error(f"Error getting session stats: {e}")
        return {
            "session_id": session_id,
            "document_count": 0,
            "chunk_count": 0
        }

# Document upload and indexing
@app.post("/upload/{session_id}")
async def upload_documents(session_id: str, files: List[UploadFile] = File(...)):
    """Upload documents for a specific session"""
    # Update session activity
    if session_id in active_sessions:
        active_sessions[session_id]["last_activity"] = datetime.now()
    
    # Create session-specific upload directory
    upload_dir = Path("./shared_uploads") / session_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    file_paths = []
    for file in files:
        # Save uploaded file
        file_id = str(uuid.uuid4())[:8]
        file_path = upload_dir / f"{file_id}_{file.filename}"
        
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        file_paths.append(str(file_path))
        logger.info(f"Uploaded file for session {session_id}: {file_path}")
    
    return {
        "message": f"Uploaded {len(files)} files",
        "file_paths": file_paths,
        "session_id": session_id
    }

@app.post("/index")
async def index_documents(request: IndexRequest):
    """Index uploaded documents for a specific session"""
    try:
        # Update session activity
        if request.session_id in active_sessions:
            active_sessions[request.session_id]["last_activity"] = datetime.now()
        
        logger.info(f"Starting indexing for session {request.session_id}")
        
        # Index with session isolation
        result = rag_pipeline.index_documents(
            request.file_paths,
            session_id=request.session_id
        )
        
        # Only proceed if indexing was successful
        if result.get("status") == "success":
            # Create ONE index record for this batch
            index_id = rag_pipeline.session_manager.create_index(
                name=f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                document_count=result.get("documents_processed", 0)
            )
            
            # Link index to session
            rag_pipeline.session_manager.link_index_to_session(request.session_id, index_id)
            
            result["index_id"] = index_id
            
            logger.info(f"✅ Indexing successful - Docs: {result['documents_processed']}, Chunks: {result['chunks_created']}")
        else:
            logger.error(f"❌ Indexing failed: {result.get('message', 'Unknown error')}")
            raise HTTPException(status_code=500, detail=result.get("message", "Indexing failed"))
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Indexing error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

# Enhanced query endpoint with MCP tools - FINAL REVISION with Working Memory Detection
@app.post("/query")
async def query_documents(request: QueryRequest):
    """Query indexed documents with MCP tools enhancement and enhanced conversation memory"""
    try:
        # Ensure session_id is provided
        if not request.session_id:
            raise HTTPException(
                status_code=400,
                detail="session_id is required. Please create a session first."
            )
        
        # Update session activity
        if request.session_id in active_sessions:
            active_sessions[request.session_id]["last_activity"] = datetime.now()

        # Extract tool flags from query
        clean_query, flag_sequential, flag_web = mcp_orchestrator.extract_tool_flags(request.query)

        # FIXED: Get FULL conversation history for context
        recent_messages = rag_pipeline.session_manager.get_session_messages(
            request.session_id, limit=12  # Get more messages for better context
        )
        
        # Format conversation history properly
        conversation_history = []
        for msg in recent_messages[-10:]:  # Last 10 messages (5 exchanges)
            role = "User" if msg["role"] == "user" else "Assistant"
            conversation_history.append(f"{role}: {msg['content']}")
        
        conversation_context = "\n".join(conversation_history) if conversation_history else "No previous conversation."

        # Detect news queries and enhance them
        if any(word in clean_query.lower() for word in ['news', 'latest', 'recent', 'update']):
            if 'latest' not in clean_query.lower() or 'news' not in clean_query.lower():
                enhanced_query = f"{clean_query} 2025"
            else:
                enhanced_query = clean_query
        else:
            enhanced_query = clean_query

        # SIMPLIFIED and AGGRESSIVE memory detection - THIS IS THE KEY FIX
        query_lower = clean_query.lower()
        
        # MEMORY DETECTION - Direct pattern matching with explicit logging
        is_memory_query = False
        memory_reason = "none"
        
        # Check each pattern explicitly
        if 'first' in query_lower and ('qns' in query_lower or 'question' in query_lower):
            is_memory_query = True
            memory_reason = "first question pattern"
        elif 'what did i ask' in query_lower:
            is_memory_query = True
            memory_reason = "what did i ask pattern"
        elif 'what was my' in query_lower:
            is_memory_query = True
            memory_reason = "what was my pattern"
        elif 'our conversation' in query_lower:
            is_memory_query = True
            memory_reason = "our conversation pattern"
        elif 'you said' in query_lower or 'you mentioned' in query_lower:
            is_memory_query = True
            memory_reason = "you said/mentioned pattern"
        elif bool(re.search(r'(explain|elaborate|tell me about|point|item)\s*\d+', query_lower)):
            is_memory_query = True
            memory_reason = "numbered reference pattern"
        elif any(phrase in query_lower for phrase in [
            'explain that', 'elaborate on that', 'tell me about that',
            'explain this', 'elaborate on this', 'tell me about this',
            'explain it', 'elaborate on it', 'tell me about it'
        ]):
            is_memory_query = True
            memory_reason = "contextual reference pattern"

        # DOCUMENT and EXTERNAL queries (only if NOT memory)
        is_document_query = False
        is_external_query = False
        is_summary_request = False
        
        if not is_memory_query:
            # Only check these if not a memory query
            document_keywords = ['document', 'summarize', 'explain this', 'both documents', 'all documents', 'both files']
            external_keywords = ['weather', 'news', 'latest', 'current', 'today']
            
            is_document_query = any(keyword in query_lower for keyword in document_keywords)
            is_external_query = any(keyword in query_lower for keyword in external_keywords)
            is_summary_request = any(word in query_lower for word in ['summarize', 'summary', 'both', 'all documents'])

        # CRITICAL: Enhanced debug logging
        logger.info(f"🔍 MEMORY DETECTION DEBUG:")
        logger.info(f"  ├─ Original query: '{clean_query}'")
        logger.info(f"  ├─ Query lower: '{query_lower}'")
        logger.info(f"  ├─ Memory detected: {is_memory_query}")
        logger.info(f"  ├─ Memory reason: {memory_reason}")
        logger.info(f"  ├─ Document query: {is_document_query}")
        logger.info(f"  ├─ External query: {is_external_query}")
        logger.info(f"  └─ Summary request: {is_summary_request}")
        
        # Additional pattern checks for debugging
        logger.info(f"🔍 PATTERN CHECKS:")
        logger.info(f"  ├─ Contains 'first': {'first' in query_lower}")
        logger.info(f"  ├─ Contains 'qns': {'qns' in query_lower}")
        logger.info(f"  ├─ Contains 'question': {'question' in query_lower}")
        logger.info(f"  ├─ Contains 'what did i ask': {'what did i ask' in query_lower}")
        logger.info(f"  └─ Contains 'you said': {'you said' in query_lower}")

        # Determine which tools to use with smart logic
        use_sequential = flag_sequential or request.use_sequential_thinking
        use_web = flag_web or request.use_web_search

        # FIXED: Smart web search logic - only use web for external queries or when no documents
        should_use_web_search = use_web and (is_external_query or is_document_query == False) and not is_memory_query

        # Apply MCP tools if enabled
        enhanced_result = None
        if use_sequential or should_use_web_search:
            # Enhance query with MCP tools
            enhanced_result = enhance_query_with_mcp_tools(
                enhanced_query,
                session_context=conversation_context,  # Pass full conversation context
                enable_sequential=use_sequential,
                enable_web_search=should_use_web_search
            )
            
            logger.info(f"Applied MCP tools: {enhanced_result['tools_used']}")

        # ENHANCED: Handle memory queries with contextual understanding
        if is_memory_query:
            logger.info(f"🧠 Processing as MEMORY QUERY - reason: {memory_reason}")
            
            # Determine if this is a numbered reference
            number_match = re.search(r'\b(\d+)\b', clean_query)
            referenced_number = number_match.group(1) if number_match else None
            
            # MEMORY QUERY: Use conversation history directly
            prompt_sections = []
            
            # Add Sequential Thinking if available
            if enhanced_result and "sequential_thinking" in enhanced_result.get("tool_outputs", {}):
                thinking = enhanced_result["tool_outputs"]["sequential_thinking"]
                prompt_sections.append("[Sequential Thinking Analysis]")
                if "approach" in thinking:
                    prompt_sections.append(f"Approach: {thinking['approach']}")
                if "thinking_steps" in thinking:
                    prompt_sections.append("\nThinking Steps:")
                    for step in thinking["thinking_steps"]:
                        prompt_sections.append(f"- {step.get('step', 'Unknown')}: {step.get('result', 'No result')}")
                prompt_sections.append("")
            
            # Add conversation history with better formatting
            prompt_sections.append("[Complete Conversation History]")
            if conversation_history:
                for i, msg in enumerate(conversation_history, 1):
                    prompt_sections.append(f"{i}. {msg}")
            else:
                prompt_sections.append("No previous conversation found in this session.")
            prompt_sections.append("")
            
            # Enhanced instructions based on query type
            if referenced_number:
                instruction_focus = f"- If the user is asking about point/item/step {referenced_number}, look for where I mentioned {referenced_number} points/items/steps in my previous responses"
            elif 'that' in query_lower or 'this' in query_lower or 'it' in query_lower:
                instruction_focus = "- If the user is referring to 'that', 'this', or 'it', look at my most recent response to understand what they're referring to"
            else:
                instruction_focus = "- If asking about 'first question' or 'first qns', find the VERY FIRST thing the user asked"
            
            final_prompt = f"""{chr(10).join(prompt_sections)}

User Question: {clean_query}

Instructions:
- Look at the COMPLETE conversation history above to answer this question
- Focus on understanding what the user is referring to based on our previous conversation
{instruction_focus}
- Do NOT search through any documents or external knowledge - use ONLY the conversation above
- Be specific and reference exactly what was said in our conversation
- If I mentioned multiple points/items and the user asks about a specific number, explain that specific point in detail
- If you cannot find the specific information in the conversation history above, say so clearly
- Answer based purely on the conversation history shown above

Answer:"""

            # Skip document retrieval for memory queries
            documents = []
            
        else:
            logger.info(f"📚 Processing as DOCUMENT/EXTERNAL QUERY")
            
            # FIXED: Smart document retrieval with balanced multi-document support
            documents = []
            if not is_external_query:
                # Use the balanced retrieval function
                documents = get_balanced_documents(
                    rag_pipeline,
                    query=clean_query,
                    session_id=request.session_id,
                    k=request.retrieval_k,
                    use_reranker=request.use_reranker
                )
            
            # Handle different query scenarios
            if is_external_query:
                # EXTERNAL QUERY: Prioritize web search, ignore documents
                if not enhanced_result or "web_search" not in enhanced_result.get("tool_outputs", {}):
                    return {
                        "answer": "This appears to be a request for current external information, but web search is not enabled. Please enable web search to get current information.",
                        "source_documents": [],
                        "session_id": request.session_id,
                        "mcp_tools_used": []
                    }
                
                web_data = enhanced_result["tool_outputs"]["web_search"]
                web_context = web_data.get("context", "No web results available")
                
                # Build web-only prompt
                prompt_sections = []
                
                # Add Sequential Thinking if available
                if enhanced_result and "sequential_thinking" in enhanced_result.get("tool_outputs", {}):
                    thinking = enhanced_result["tool_outputs"]["sequential_thinking"]
                    prompt_sections.append("[Sequential Thinking Analysis]")
                    if "approach" in thinking:
                        prompt_sections.append(f"Approach: {thinking['approach']}")
                    if "thinking_steps" in thinking:
                        prompt_sections.append("\nThinking Steps:")
                        for step in thinking["thinking_steps"]:
                            prompt_sections.append(f"- {step.get('step', 'Unknown')}: {step.get('result', 'No result')}")
                    prompt_sections.append("")
                
                # Add web search results
                prompt_sections.append("[Current Web Information]")
                if web_data.get("results"):
                    for i, result in enumerate(web_data["results"][:5], 1):
                        prompt_sections.append(f"{i}. {result.get('title', 'No title')}")
                        prompt_sections.append(f"   {result.get('snippet', 'No snippet')[:300]}")
                        prompt_sections.append(f"   Source: {result.get('displayLink', 'Unknown')}")
                        prompt_sections.append("")
                
                final_prompt = f"""{chr(10).join(prompt_sections)}

User Question: {clean_query}

Instructions:
- Use ONLY the current web information provided above
- Do NOT reference any uploaded documents or internal knowledge
- Provide accurate, up-to-date information based on web sources
- If web information is insufficient, clearly state what's missing
- Cite sources when providing specific facts

Answer:"""
                
            elif is_document_query and documents:
                # DOCUMENT QUERY: Use documents with improved multi-document context building
                
                # IMPROVED: Build context ensuring all documents are represented
                doc_groups = {}
                for doc in documents:
                    doc_id = doc.get("document_id", "unknown")
                    if doc_id not in doc_groups:
                        doc_groups[doc_id] = []
                    doc_groups[doc_id].append(doc["text"])
                
                # Build context parts ensuring representation from all documents
                context_parts = []
                for doc_id, chunks in doc_groups.items():
                    context_parts.append(f"[Document ID: {doc_id}]")
                    # Take top chunks from each document
                    context_parts.extend(chunks[:8])  # More chunks per document for better coverage
                    context_parts.append("")
                
                doc_context = "\n\n".join(context_parts)
                
                prompt_sections = []
                
                # Add Sequential Thinking if available
                if enhanced_result and "sequential_thinking" in enhanced_result.get("tool_outputs", {}):
                    thinking = enhanced_result["tool_outputs"]["sequential_thinking"]
                    prompt_sections.append("[Sequential Thinking Analysis]")
                    if "approach" in thinking:
                        prompt_sections.append(f"Approach: {thinking['approach']}")
                    if "thinking_steps" in thinking:
                        prompt_sections.append("\nThinking Steps:")
                        for step in thinking["thinking_steps"]:
                            prompt_sections.append(f"- {step.get('step', 'Unknown')}: {step.get('result', 'No result')}")
                    prompt_sections.append("")
                
                thinking_context = "\n".join(prompt_sections) if prompt_sections else ""
                
                # ENHANCED: Multi-document summarization prompt
                if is_summary_request:
                    final_prompt = f"""{thinking_context}

Document Context:
{doc_context}

User Question: {clean_query}

Instructions:
- Provide a comprehensive summary that covers ALL documents provided above
- Ensure you include key points from EACH document (note the Document ID labels)
- Create a balanced summary that represents all uploaded documents
- If you notice content from only one document, explicitly mention this limitation
- Structure your summary to clearly show insights from multiple sources
- Reference document IDs when highlighting specific points

Answer:"""
                else:
                    final_prompt = f"""{thinking_context}

Document Context:
{doc_context}

User Question: {clean_query}

Instructions:
- Provide a comprehensive answer based on ALL document content above
- Be specific and reference key points from the documents
- When relevant, mention which document (by ID) contains specific information
- Follow any analysis framework provided above
- If information is insufficient, clearly state what's missing

Answer:"""
                
            else:
                # GENERAL/MIXED QUERY: Handle lack of documents or hybrid needs
                if not documents and not enhanced_result:
                    return {
                        "answer": "No documents found in this session. Please upload and index documents first, or enable web search for external information.",
                        "source_documents": [],
                        "session_id": request.session_id,
                        "mcp_tools_used": []
                    }
                
                # Build context from available sources
                context_parts = []
                
                # ADDED: Include conversation context for general queries
                if conversation_history and not is_external_query:
                    context_parts.append("[Previous Conversation]")
                    context_parts.extend(conversation_history[-4:])  # Last 4 messages
                    context_parts.append("")
                
                # Add Sequential Thinking if available
                if enhanced_result and "sequential_thinking" in enhanced_result.get("tool_outputs", {}):
                    thinking = enhanced_result["tool_outputs"]["sequential_thinking"]
                    context_parts.append("[Sequential Thinking Analysis]")
                    if "approach" in thinking:
                        context_parts.append(f"Approach: {thinking['approach']}")
                    if "thinking_steps" in thinking:
                        context_parts.append("\nThinking Steps:")
                        for step in thinking["thinking_steps"]:
                            context_parts.append(f"- {step.get('step', 'Unknown')}: {step.get('result', 'No result')}")
                    context_parts.append("")
                
                # Add document context if available with multi-document support
                if documents:
                    doc_groups = {}
                    for doc in documents:
                        doc_id = doc.get("document_id", "unknown")
                        if doc_id not in doc_groups:
                            doc_groups[doc_id] = []
                        doc_groups[doc_id].append(doc["text"])
                    
                    context_parts.append("[Document Context]")
                    for doc_id, chunks in doc_groups.items():
                        context_parts.append(f"Document {doc_id}:")
                        context_parts.extend(chunks[:5])  # Top 5 chunks per document
                    context_parts.append("")
                
                # Add web context if available
                if enhanced_result and "web_search" in enhanced_result.get("tool_outputs", {}):
                    web_data = enhanced_result["tool_outputs"]["web_search"]
                    if web_data.get("results"):
                        context_parts.append("[Current Web Information]")
                        for i, result in enumerate(web_data["results"][:3], 1):
                            context_parts.append(f"{i}. {result.get('title', 'No title')}")
                            context_parts.append(f"   {result.get('snippet', 'No snippet')[:200]}")
                            context_parts.append(f"   Source: {result.get('displayLink', 'Unknown')}")
                        context_parts.append("")
                
                combined_context = "\n".join(context_parts) if context_parts else "No specific context available."
                
                final_prompt = f"""{combined_context}

User Question: {clean_query}

Instructions:
- Use the most relevant context above to answer the question
- If the question requires current information, prioritize web sources
- If the question is about uploaded documents, prioritize document sources
- If the question is about previous conversation, use the conversation history
- Be specific and cite your sources (including document IDs when relevant)
- If information is insufficient, clearly state what's missing

Answer:"""
        
        logger.info(f"Sending prompt to Ollama (length: {len(final_prompt)} chars)")
        logger.debug(f"Query type - Document: {is_document_query}, External: {is_external_query}, Memory: {is_memory_query}, Summary: {is_summary_request}")
        
        # Generate response with explicit model specification
        answer = ollama_client.generate(final_prompt, model="qwen2:7b")
        
        # Check if we got a valid response
        if not answer or answer.strip() == "" or answer.startswith("Error"):
            logger.error(f"No valid response from Ollama: {answer}")
            answer = "I apologize, but I couldn't generate a proper response. Please try again."
        else:
            logger.info(f"Successfully generated response: {len(answer)} chars")
        
        # Format MCP tools indicator for response
        tools_used_text = ""
        if enhanced_result and enhanced_result.get("tools_used"):
            tools_indicators = []
            if "Sequential Thinking" in enhanced_result["tools_used"]:
                tools_indicators.append("🧠 Sequential Thinking")
            if "Web Search" in enhanced_result["tools_used"]:
                tools_indicators.append("🔍 Web Search")
            
            if tools_indicators:
                tools_used_text = f"[{' • '.join(tools_indicators)}]\n\n"
        
        final_answer = answer
        
        # Save to session
        rag_pipeline.session_manager.add_message(request.session_id, "user", clean_query)
        rag_pipeline.session_manager.add_message(request.session_id, "assistant", final_answer)
        
        # Prepare response with enhanced debug info
        response_data = {
            "answer": final_answer,
            "source_documents": documents[:10] if documents and not is_memory_query else [],  # Don't show docs for memory queries
            "session_id": request.session_id,
            "mcp_tools_used": enhanced_result.get("tools_used", []) if enhanced_result else [],
            "documents_used": len(set(doc.get("document_id") for doc in documents)) if documents and not is_memory_query else 0,
            "conversation_context_used": len(conversation_history) > 0,  # NEW: Indicate if conversation context was used
            "query_type": "memory" if is_memory_query else ("external" if is_external_query else ("document" if is_document_query else "general")),  # NEW: Debug info
            "debug_info": {  # ENHANCED: More debug information
                "memory_detected": is_memory_query,
                "memory_reason": memory_reason,
                "query_lower": query_lower,
                "contains_first": 'first' in query_lower,
                "contains_qns": 'qns' in query_lower,
                "contains_question": 'question' in query_lower,
                "contains_what_did_i_ask": 'what did i ask' in query_lower,
                "conversation_history_length": len(conversation_history)
            }
        }
        
        # Add web search results if available
        if enhanced_result and "web_search" in enhanced_result.get("tool_outputs", {}):
            response_data["web_sources"] = enhanced_result["tool_outputs"]["web_search"].get("results", [])
        
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
async def chat(request: ChatRequest):
    """Chat endpoint with MCP tools support"""
    try:
        # Ensure session_id
        if not request.session_id:
            session_response = await create_session()
            request.session_id = session_response["session_id"]
        
        # Update session activity
        if request.session_id in active_sessions:
            active_sessions[request.session_id]["last_activity"] = datetime.now()
        
        # Get the latest message
        latest_message = request.messages[-1] if request.messages else None
        if not latest_message:
            raise HTTPException(status_code=400, detail="No messages provided")
        
        # Apply MCP tools if enabled
        enhanced_content = latest_message.content
        tools_info = ""
        
        if request.use_sequential_thinking or request.use_web_search:
            enhanced_result = enhance_query_with_mcp_tools(
                latest_message.content,
                enable_sequential=request.use_sequential_thinking,
                enable_web_search=request.use_web_search
            )
            
            if enhanced_result["tools_used"]:
                tools_indicators = []
                if "Sequential Thinking" in enhanced_result["tools_used"]:
                    tools_indicators.append("🧠 Sequential Thinking")
                if "Web Search" in enhanced_result["tools_used"]:
                    tools_indicators.append("🔍 Web Search")
                tools_info = f"[{' • '.join(tools_indicators)}]\n\n"
            
            # Add tool context to the message
            if "sequential_thinking" in enhanced_result["tool_outputs"]:
                thinking = enhanced_result["tool_outputs"]["sequential_thinking"]
                enhanced_content += f"\n\n[Approach: {thinking['approach']}]"
            
            if "web_search" in enhanced_result["tool_outputs"]:
                web_context = enhanced_result["tool_outputs"]["web_search"]["context"]
                enhanced_content += f"\n\n{web_context}"
        
        # Convert messages to Ollama format with enhancement
        messages = []
        
        # Add system message
        messages.append({
            "role": "system",
            "content": "You are a helpful assistant with access to documents and advanced reasoning capabilities. You can only access documents uploaded in this specific session."
        })
        
        # Add conversation history
        for msg in request.messages[:-1]:
            messages.append({"role": msg.role, "content": msg.content})
        
        # Add the enhanced latest message
        messages.append({"role": latest_message.role, "content": enhanced_content})
        
        # Get response from Ollama
        response = ollama_client.chat(messages, model=request.model or config.GENERATION_MODEL)
        
        # Add tools indicator to response
        final_response = tools_info + response
        
        # Save to session
        rag_pipeline.session_manager.add_message(
            request.session_id, 
            latest_message.role, 
            latest_message.content
        )
        rag_pipeline.session_manager.add_message(
            request.session_id, 
            "assistant", 
            final_response
        )
        
        return {
            "response": final_response, 
            "session_id": request.session_id,
            "mcp_tools_used": enhanced_result["tools_used"] if request.use_sequential_thinking or request.use_web_search else []
        }
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)