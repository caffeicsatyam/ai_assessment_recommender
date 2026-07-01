# Use slim Python base for smaller image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed by faiss-cpu
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Copy pre-built data files (catalog + FAISS index)
COPY data/ ./data/

# Expose port (Cloud Run uses 8080 by default)
EXPOSE 8080

# Start the FastAPI server
# Cloud Run injects PORT env variable; default to 8080
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
