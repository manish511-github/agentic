FROM python:3.10-slim
   # Install system dependencies
RUN apt-get update && apt-get install -y \
wget \
gnupg \
unzip \
curl \
chromium \
&& rm -rf /var/lib/apt/lists/*


# Set working directory
WORKDIR /app
ENV PYTHONPATH=/app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and Alembic migrations
COPY . .

# Expose port
EXPOSE 8000

# Command to run migrations and app
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000"]