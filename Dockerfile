FROM python:3.11-slim

WORKDIR /app

# Build tools required by scientific/quantum Python packages.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Cloud Run deployment is CPU-only. Install the CPU PyTorch wheel so CUDA
# libraries are not pulled into the image.
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Fail the image build early if the core quantum stack cannot import.
RUN python -c "import torch; import autoray; import pennylane; print('Torch:', torch.__version__); print('Autoray:', autoray.__version__); print('PennyLane:', pennylane.__version__); print('PennyLane import: OK')"

COPY app.py .
COPY API_files/ ./API_files/

ENV PORT=8080
ENV PYTHONUNBUFFERED=1

EXPOSE 8080

CMD ["python", "app.py"]
