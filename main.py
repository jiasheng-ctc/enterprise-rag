"""
Updated Main entry point for Enterprise RAG System - Docker optimized
"""
import subprocess
import sys
import signal
import time
from pathlib import Path
import logging
import threading
import os
import asyncio

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DockerSystemManager:
    """Manages all system components for Docker deployment"""
    
    def __init__(self):
        self.processes = []
        self.threads = []
        self.is_docker = os.path.exists('/.dockerenv')
        
    def stream_output(self, process, name, color_code):
        """Stream output from a process with colored prefix"""
        colors = {
            'backend': '\033[92m',   # Green
            'frontend': '\033[93m',  # Yellow
            'reset': '\033[0m'
        }
        
        color = colors.get(color_code, colors['reset'])
        
        # Stream stdout
        try:
            for line in iter(process.stdout.readline, b''):
                if line:
                    print(f"{color}[{name}]{colors['reset']} {line.decode('utf-8').rstrip()}")
        except Exception as e:
            logger.error(f"Error streaming {name} output: {e}")
    
    def start_backend(self):
        """Start FastAPI backend"""
        logger.info("Starting backend server...")
        
        # Set environment variables for backend
        env = os.environ.copy()
        env['PYTHONPATH'] = '/app' if self.is_docker else '.'
        
        process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "rag_system.api.server:app", 
             "--host", "0.0.0.0", "--port", "8000"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=False,
            env=env
        )
        self.processes.append(process)
        
        # Start thread to stream output
        thread = threading.Thread(
            target=self.stream_output,
            args=(process, "BACKEND", "backend")
        )
        thread.daemon = True
        thread.start()
        self.threads.append(thread)
        
        logger.info("Backend server starting on port 8000...")
    
    def start_frontend(self):
        """Start Dash frontend"""
        logger.info("Starting frontend...")
        
        # Check if frontend app exists
        frontend_path = Path("frontend/app.py")
        if not frontend_path.exists():
            logger.error("Frontend app.py not found in ./frontend/")
            return
        
        # Set environment variables for frontend
        env = os.environ.copy()
        env['PYTHONPATH'] = '/app' if self.is_docker else '.'
        env['API_URL'] = 'http://localhost:8000'
        
        process = subprocess.Popen(
            [sys.executable, "app.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd="frontend",
            bufsize=1,
            universal_newlines=False,
            env=env
        )
        self.processes.append(process)
        
        # Start thread to stream output
        thread = threading.Thread(
            target=self.stream_output,
            args=(process, "FRONTEND", "frontend")
        )
        thread.daemon = True
        thread.start()
        self.threads.append(thread)
        
        logger.info("Frontend starting on port 3000...")
    
    def wait_for_service(self, url, service_name, max_attempts=60):
        """Wait for a service to be ready"""
        import requests
        for i in range(max_attempts):
            try:
                response = requests.get(url, timeout=2)
                if response.status_code in [200, 404]:  # 404 is OK for frontend root
                    logger.info(f"✓ {service_name} is ready!")
                    return True
            except:
                pass
            
            if i % 10 == 0 and i > 0:
                logger.info(f"Still waiting for {service_name}... ({i}/{max_attempts})")
            time.sleep(1)
        
        logger.warning(f"Service {service_name} did not become ready in time")
        return False
    
    def wait_for_ollama(self, max_attempts=30):
        """Wait for Ollama to be ready"""
        import requests
        ollama_url = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
        
        logger.info(f"Waiting for Ollama at {ollama_url}...")
        
        for i in range(max_attempts):
            try:
                response = requests.get(f"{ollama_url}/api/tags", timeout=5)
                if response.status_code == 200:
                    logger.info("✓ Ollama is ready!")
                    return True
            except Exception as e:
                if i % 5 == 0:
                    logger.info(f"Waiting for Ollama... ({i}/{max_attempts}) - {e}")
            time.sleep(2)
        
        logger.warning("Ollama did not become ready in time")
        return False
    
    def stop_all(self):
        """Stop all processes"""
        logger.info("Stopping all services...")
        for process in self.processes:
            try:
                process.terminate()
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning("Process didn't terminate gracefully, killing...")
                process.kill()
            except Exception as e:
                logger.error(f"Error stopping process: {e}")
        
        logger.info("All services stopped")
    
    def run(self):
        """Run the complete system"""
        # Set up signal handler for graceful shutdown
        signal.signal(signal.SIGINT, lambda sig, frame: self.shutdown())
        signal.signal(signal.SIGTERM, lambda sig, frame: self.shutdown())
        
        try:
            # Wait for Ollama if in Docker
            if self.is_docker:
                self.wait_for_ollama()
            
            # Start Backend
            self.start_backend()
            
            # Wait for backend to be ready
            if self.wait_for_service("http://localhost:8000/health", "Backend API"):
                logger.info("✓ Backend is healthy")
            
            # Start Frontend
            self.start_frontend()
            
            # Wait a bit for frontend to start
            time.sleep(3)
            
            # Print access information
            print("\n" + "="*70)
            print("🚀 Enterprise RAG System Started Successfully!")
            print("="*70)
            
            if self.is_docker:
                print("🐳 Running in Docker container")
                print("\n📍 Access Points (from host machine):")
                print("  • Frontend UI:  http://localhost:3000")
                print("  • API Docs:     http://localhost:8000/docs")
                print("  • Health Check: http://localhost:8000/health")
                print("  • Ollama:       http://localhost:11434")
            else:
                print("💻 Running locally")
                print("\n📍 Access Points:")
                print("  • Frontend UI:  http://localhost:3000")
                print("  • API Docs:     http://localhost:8000/docs")
                print("  • Health Check: http://localhost:8000/health")
            
            print("\n📝 Features Enabled:")
            print("  • Session-based document isolation")
            print("  • Sequential Thinking for complex reasoning")
            print("  • Web Search for current information")
            print("  • Auto-cleanup of inactive sessions")
            
            print("\n📋 Logs are displayed below in real-time")
            print("🛑 Press Ctrl+C to stop all services")
            print("="*70 + "\n")
            
            # Keep running and monitor processes
            while True:
                # Check if any process has died
                dead_processes = [p for p in self.processes if p.poll() is not None]
                if dead_processes:
                    logger.error(f"{len(dead_processes)} process(es) have died. Shutting down...")
                    break
                
                time.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        except Exception as e:
            logger.error(f"Error during startup: {e}")
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Graceful shutdown"""
        print("\n" + "="*70)
        logger.info("🛑 Shutting down Enterprise RAG System...")
        self.stop_all()
        print("="*70)
        sys.exit(0)

def check_environment():
    """Check if we have the required environment"""
    logger.info("Checking environment...")
    
    # Check if we're in Docker
    is_docker = os.path.exists('/.dockerenv')
    if is_docker:
        logger.info("Running in Docker container")
    else:
        logger.info("Running locally")
    
    # Check required directories
    required_dirs = ["rag_system", "frontend"]
    for dir_name in required_dirs:
        if not Path(dir_name).exists():
            logger.error(f"Required directory not found: {dir_name}")
            sys.exit(1)
    
    # Check if main modules can be imported
    try:
        import rag_system.api.server
        logger.info("✓ Backend modules can be imported")
    except ImportError as e:
        logger.error(f"Cannot import backend modules: {e}")
        sys.exit(1)
    
    # Check if frontend exists
    if not Path("frontend/app.py").exists():
        logger.error("Frontend app.py not found")
        sys.exit(1)
    
    logger.info("✓ Environment check passed")

def main():
    """Main entry point"""
    print("🚀 Starting Enterprise RAG System...")
    
    # Check environment first
    check_environment()
    
    # Create and run system manager
    manager = DockerSystemManager()
    manager.run()

if __name__ == "__main__":
    main()