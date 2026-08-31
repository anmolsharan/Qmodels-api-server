FROM python:3.11-slim

WORKDIR /app

# Build tools for scientific/quantum Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first.
# Cloud Run does not provide a GPU in this configuration.
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    --index-url https://download.pytorch.org/whl/cpu

# Copy Python dependency list
COPY requirements.txt .

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# IMPORTANT:
# Verify the PennyLane/Autoray combination during image build.
# If this fails, Cloud Build will fail here instead of producing
# a broken Cloud Run image.
RUN python -c "import torch; import autoray; import pennylane; print('Torch:', torch.__version__); print('Autoray:', autoray.__version__); print('PennyLane:', pennylane.__version__); print('PennyLane import: OK')"

# Copy application
COPY app.py .

# Copy model/data files
COPY API_files/ ./API_files/

# Cloud Run listens on PORT
ENV PORT=8080

EXPOSE 8080

CMD ["python", "app.py"]