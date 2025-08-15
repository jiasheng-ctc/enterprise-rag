"""
Simplified Ollama Client for qwen2:7b
"""
import requests
import json
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

class OllamaClient:
    """Simple client for Ollama API"""
    
    def __init__(self, host: str = "http://localhost:11434"):
        self.host = host
        self.api_url = f"{host}/api"
        self.default_model = "qwen2:7b"  # Your model
    
    def is_running(self) -> bool:
        """Check if Ollama is running"""
        try:
            response = requests.get(f"{self.api_url}/tags", timeout=5)
            if response.status_code == 200:
                logger.info("Ollama is running")
                return True
            return False
        except Exception as e:
            logger.error(f"Ollama not accessible: {e}")
            return False
    
    def generate(self, prompt: str, model: str = None, stream: bool = False) -> str:
        """Generate text completion"""
        if model is None:
            model = self.default_model
            
        try:
            # Log the prompt for debugging
            logger.info(f"Generating response with {model}")
            logger.debug(f"Prompt length: {len(prompt)} characters")
            
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,  # Force non-streaming
                "options": {
                    "temperature": 0.7,
                    "num_predict": 2048,  # Adequate for summaries
                    "top_k": 40,
                    "top_p": 0.9
                }
            }
            
            response = requests.post(
                f"{self.api_url}/generate",
                json=payload,
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get("response", "")
                
                if generated_text:
                    logger.info(f"Generated {len(generated_text)} characters")
                    return generated_text
                else:
                    logger.warning("Empty response from Ollama")
                    return "I couldn't generate a response. Please try again."
            else:
                logger.error(f"Ollama error: {response.status_code} - {response.text}")
                return f"Error generating response (status: {response.status_code})"
                
        except requests.exceptions.Timeout:
            logger.error("Request timed out")
            return "Request timed out. Please try with a shorter prompt."
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return f"Error: {str(e)}"
    
    def chat(self, messages: List[Dict[str, str]], model: str = None) -> str:
        """Chat completion"""
        if model is None:
            model = self.default_model
            
        try:
            payload = {
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 2048
                }
            }
            
            response = requests.post(
                f"{self.api_url}/chat",
                json=payload,
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                message = result.get("message", {})
                return message.get("content", "")
            else:
                logger.error(f"Chat error: {response.status_code}")
                return "Error in chat completion"
                
        except Exception as e:
            logger.error(f"Chat error: {e}")
            return f"Error: {str(e)}"