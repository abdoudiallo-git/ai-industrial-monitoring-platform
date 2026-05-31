# Base image
FROM python:3.12-slim

# Working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# Copy project files
COPY src/ ./src/
COPY api/ ./api/
COPY models/ ./models/
COPY data/processed/ ./data/processed/

# Expose port
EXPOSE 8000

# Run FastAPI
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]