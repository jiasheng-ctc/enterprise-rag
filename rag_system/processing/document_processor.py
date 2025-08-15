import os
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Enhanced document processor for PDF and DOCX files"""
    
    SUPPORTED_FORMATS = {'.pdf', '.docx', '.txt'}
    
    def __init__(self):
        self.processed_count = 0
        
    def process_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Process a single document file"""
        path = Path(file_path)
        
        if not path.exists():
            logger.error(f"File not found: {file_path}")
            return None
            
        if path.suffix.lower() not in self.SUPPORTED_FORMATS:
            logger.warning(f"Unsupported format: {path.suffix}")
            return None
            
        try:
            content = self._extract_content(path)
            
            # Log extraction result
            if content:
                logger.info(f"✅ Extracted {len(content)} characters from {path.name}")
            else:
                logger.warning(f"⚠️ No content extracted from {path.name}")
            
            if content and len(content.strip()) > 0:
                self.processed_count += 1
                return {
                    "document_id": path.stem,
                    "filename": path.name,
                    "content": content,
                    "metadata": {
                        "file_path": str(path),
                        "file_size": path.stat().st_size,
                        "file_type": path.suffix.lower(),
                        "content_length": len(content)
                    }
                }
            else:
                logger.error(f"Document {path.name} has no extractable content")
                return None
                
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _extract_content(self, path: Path) -> str:
        """Extract text content from file"""
        if path.suffix.lower() == '.pdf':
            return self._extract_pdf_enhanced(path)
        elif path.suffix.lower() == '.docx':
            return self._extract_docx(path)
        elif path.suffix.lower() == '.txt':
            return path.read_text(encoding='utf-8', errors='ignore')
        return ""
    
    def _extract_pdf_enhanced(self, path: Path) -> str:
        """Enhanced PDF extraction with multiple methods"""
        content = ""
        
        # Method 1: Try PyPDF2 first
        try:
            import PyPDF2
            text_parts = []
            with open(path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                logger.info(f"PDF has {num_pages} pages")
                
                for page_num in range(num_pages):
                    try:
                        page = pdf_reader.pages[page_num]
                        page_text = page.extract_text()
                        if page_text:
                            text_parts.append(page_text)
                            logger.debug(f"Page {page_num + 1}: extracted {len(page_text)} chars")
                    except Exception as e:
                        logger.warning(f"Failed to extract page {page_num + 1}: {e}")
                
                content = "\n".join(text_parts)
                
                if content.strip():
                    logger.info(f"✅ PyPDF2 extracted {len(content)} characters")
                    return content
                    
        except Exception as e:
            logger.warning(f"PyPDF2 extraction failed: {e}")
        
        # Method 2: Try pdfplumber as fallback
        try:
            import pdfplumber
            text_parts = []
            with pdfplumber.open(path) as pdf:
                for i, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
                        logger.debug(f"Page {i + 1}: extracted {len(page_text)} chars with pdfplumber")
            
            content = "\n".join(text_parts)
            
            if content.strip():
                logger.info(f"✅ pdfplumber extracted {len(content)} characters")
                return content
                
        except ImportError:
            logger.info("pdfplumber not installed, skipping this method")
        except Exception as e:
            logger.warning(f"pdfplumber extraction failed: {e}")
        
        # Method 3: Try PyMuPDF (fitz) as another fallback
        try:
            import fitz  # PyMuPDF
            text_parts = []
            doc = fitz.open(str(path))
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_text = page.get_text()
                if page_text:
                    text_parts.append(page_text)
                    
            doc.close()
            content = "\n".join(text_parts)
            
            if content.strip():
                logger.info(f"✅ PyMuPDF extracted {len(content)} characters")
                return content
                
        except ImportError:
            logger.info("PyMuPDF not installed, skipping this method")
        except Exception as e:
            logger.warning(f"PyMuPDF extraction failed: {e}")
        
        # If all methods fail, log error
        logger.error(f"❌ All PDF extraction methods failed for {path.name}")
        return ""
    
    def _extract_docx(self, path: Path) -> str:
        """Extract text from DOCX"""
        try:
            import docx
            doc = docx.Document(str(path))
            paragraphs = []
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    paragraphs.append(paragraph.text)
            
            content = "\n".join(paragraphs)
            logger.info(f"Extracted {len(content)} characters from DOCX")
            return content
        except Exception as e:
            logger.error(f"DOCX extraction failed: {e}")
            return ""
    
    def process_batch(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """Process multiple documents"""
        results = []
        for file_path in file_paths:
            logger.info(f"Processing file: {file_path}")
            result = self.process_file(file_path)
            if result:
                results.append(result)
                logger.info(f"✅ Successfully processed: {result['filename']}")
            else:
                logger.error(f"❌ Failed to process: {file_path}")
        return results