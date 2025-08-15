# Multi-stage build for smaller image
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt && \
    pip install --no-cache-dir --user \
    duckduckgo-search \
    python-dotenv \
    dash \
    dash-bootstrap-components \
    dash-mantine-components

# Production stage
FROM python:3.11-slim

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Make sure scripts in .local are available
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY . .

# Create necessary directories with proper permissions
RUN mkdir -p /app/shared_uploads \
    /app/index_store/lancedb \
    /app/logs \
    /app/backend \
    /app/frontend/assets && \
    chmod -R 755 /app

# Set environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2 \
    GENERATION_MODEL=qwen2:7b \
    RERANKER_MODEL=BAAI/bge-reranker-base \
    OLLAMA_HOST=http://localhost:11434 \
    LANCEDB_PATH=/app/index_store/lancedb \
    SESSION_DB_PATH=/app/backend/chat_data.db \
    CHUNK_SIZE=512 \
    CHUNK_OVERLAP=50 \
    EMBEDDING_BATCH_SIZE=50 \
    BACKEND_PORT=8000 \
    FRONTEND_PORT=3000

# Expose ports
EXPOSE 8000 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health && curl -f http://localhost:3000/ || exit 1

# Start the application
CMD ["python", "main.py"]