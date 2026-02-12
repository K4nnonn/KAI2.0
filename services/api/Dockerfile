FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY services/api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY services/api/kai_core/ ./kai_core/
COPY services/api/utils/ ./utils/
COPY services/api/static/ ./static/
COPY services/api/*.py ./
COPY services/worker/jobs_worker.py ./jobs_worker.py

# Set environment variables
ENV PYTHONPATH=/app
ENV AZURE_CONFIG_DIR=/app/.azure

# Expose port
EXPOSE 8000

# Default worker count (override in container app)
ENV UVICORN_WORKERS=2

# Run FastAPI with uvicorn
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port 8000 --workers ${UVICORN_WORKERS:-2}"]
