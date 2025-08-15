#!/usr/bin/env python3
"""Test script to verify system setup"""

import sys
import requests
from pathlib import Path

def test_imports():
    """Test if all modules can be imported"""
    try:
        from rag_system.core.rag_pipeline import RAGPipeline
        from rag_system.utils.ollama_client import OllamaClient
        from rag_system.config.settings import config
        print("✓ All modules imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_ollama():
    """Test Ollama connection"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            print(f"✓ Ollama is running with {len(models)} models")
            return True
        else:
            print("✗ Ollama is not responding correctly")
            return False
    except Exception as e:
        print(f"✗ Cannot connect to Ollama: {e}")
        return False

def test_directories():
    """Test if required directories exist"""
    dirs = ["shared_uploads", "index_store/lancedb", "backend"]
    for dir_path in dirs:
        if Path(dir_path).exists():
            print(f"✓ Directory exists: {dir_path}")
        else:
            print(f"✗ Directory missing: {dir_path}")
            return False
    return True

if __name__ == "__main__":
    print("=" * 50)
    print("Enterprise RAG System Setup Test")
    print("=" * 50)
    
    all_good = True
    all_good = test_imports() and all_good
    all_good = test_ollama() and all_good
    all_good = test_directories() and all_good
    
    print("=" * 50)
    if all_good:
        print("✅ System setup completed successfully!")
        print("\nYou can now:")
        print("1. Start the API server: python -m uvicorn rag_system.api.server:app --reload")
        print("2. Or use the main script: python main.py")
        print("3. Access API docs at: http://localhost:8000/docs")
    else:
        print("❌ Setup incomplete. Please fix the issues above.")
        sys.exit(1)
