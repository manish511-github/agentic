FROM python:3.10-slim
   # Install system dependencies
RUN apt-get update && apt-get install -y \
wget \
gnupg \
unzip \
curl \
chromium \
gcc \
g++ \
build-essential \
software-properties-common \
git \
&& rm -rf /var/lib/apt/lists/*


# Set working directory
WORKDIR /app
ENV PYTHONPATH=/app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir watchgod

# Commented out SentenceTransformer model pre-download
# Pre-download the SentenceTransformer model
# RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2', cache_folder='/app/model_cache')"

# Copy application code and Alembic migrations
COPY . .

# Create necessary directories
# Commented out model_cache directory (was used for SentenceTransformer)
# RUN mkdir -p /app/model_cache /app/cookies
RUN mkdir -p /app/cookies

# Expose port
EXPOSE 8000

# Command to run migrations and app with auto-reload
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --reload-dir /app/app"]