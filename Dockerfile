FROM python:3.11-slim

WORKDIR /app

# build-essential is here defensively — some scientific packages fall back to
# building from source if a prebuilt wheel isn't available for the platform.
# Safe to try removing later if your build works fine without it (smaller image).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install torch CPU-only build first, from the CPU wheel index — this avoids
# pulling in ~2GB of CUDA libraries you don't need (your server confirmed
# CUDA is not available, and Cloud Run has no GPU by default either).
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY API_files/ ./API_files/

# Cloud Run sets PORT itself and expects the container to listen on it;
# 8080 here is just the local default for `docker run` without that env var.
ENV PORT=8080
EXPOSE 8080

CMD ["python", "app.py"]
