# Base Image with Python 3.11
FROM python:3.11-slim

# Set environment variables for non-interactive execution & unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860

WORKDIR /app

# Install system dependencies (build-essential, git)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code, data, and dashboard UI
COPY backend /app/backend
COPY dashboard /app/dashboard
COPY data /app/data

# Ensure data directory exists and is writable
RUN mkdir -p /app/backend/data /app/data && chmod -R 777 /app/backend/data /app/data

WORKDIR /app/backend

# Expose Hugging Face Space default port 7860
EXPOSE 7860

# Run uvicorn server binding to 0.0.0.0:7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
