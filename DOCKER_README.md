# Quantum Models API — Docker / Google Cloud deployment

## 1. Project layout

```
quantum_api_docker/
├── app.py                  # the API (converted from your notebook — see notes below)
├── requirements.txt
├── Dockerfile
├── .dockerignore
└── API_files/               # <-- YOU NEED TO POPULATE THIS before building
    ├── artifacts_50/
    │   ├── rx_encoding_weights.pt
    │   ├── ry_encoding_weights.pt
    │   ├── rz_encoding_weights.pt
    │   ├── learnable_encoding_weights.pt
    │   └── data_re-uploading_weights.pt
    ├── X_Test_rfe_new.csv
    ├── y_Test_rfe_new.csv
    ├── X_Test_QSVC_38.csv
    ├── y_Test_QSVC_38.csv
    ├── X_train_norm_38.npy
    ├── qsvc_38.pkl
    ├── X_Test_QKNN.csv
    ├── y_Test_QKNN.csv
    └── qknn_model.pkl
```

**Before building**, copy your actual model/data files from
`/home/ankit/QAPIs/API_files/` on your server into this `API_files/` folder,
preserving the same relative structure. Total size should be well under
100MB given the file sizes you showed earlier (`qknn_model.pkl` at 6.27MB is
the largest), so baking them directly into the image is simpler than mounting
external storage.

```bash
# from your server, or wherever the files currently live:
scp -r ankit@your-server:/home/ankit/QAPIs/API_files/artifacts_50 ./API_files/
scp ankit@your-server:/home/ankit/QAPIs/API_files/{X_Test_rfe_new.csv,y_Test_rfe_new.csv,X_Test_QSVC_38.csv,y_Test_QSVC_38.csv,X_train_norm_38.npy,qsvc_38.pkl,X_Test_QKNN.csv,y_Test_QKNN.csv,qknn_model.pkl} ./API_files/
```

## 2. What changed from the notebook

- Removed `nest_asyncio` / `await server.serve()` (that pattern only works
  inside a running Jupyter event loop) — replaced with a plain
  `if __name__ == "__main__": uvicorn.run(...)` block.
- The app now listens on `$PORT` (Cloud Run injects this automatically;
  defaults to `8080` for local `docker run`) instead of the hardcoded `6000`.
- All file paths now resolve through `QAPI_FILES_DIR`, defaulting to
  `/app/API_files` inside the container — override with
  `-e QAPI_FILES_DIR=/some/other/path` if you later switch to mounted
  external storage instead of files baked into the image.
- Everything else — the VQC parallel `ProcessPoolExecutor`, the QSVC/QKNN
  vectorized fidelity math, endpoint logic, response shapes — is untouched.

## 3. Build and test locally first

```bash
docker build -t quantum-api .
docker run --rm -p 8080:8080 quantum-api
```

Then hit it the same way you would in Postman:
```bash
curl -X POST http://localhost:8080/qknn/classification-metrics \
  -H "Content-Type: application/json" \
  -d '{"features": [...your 44 feature names...]}'
```

## 4. Push to Google Cloud

```bash
export PROJECT_ID=your-gcp-project-id
export REGION=us-central1          # pick your region
export REPO=quantum-api-repo

# One-time: create an Artifact Registry repo
gcloud artifacts repositories create $REPO \
  --repository-format=docker \
  --location=$REGION

# Build and push (Cloud Build does this without needing local Docker)
gcloud builds submit --tag $REGION-docker.pkg.dev/$PROJECT_ID/$REPO/quantum-api:latest
```

## 5. Deploy — two options depending on how many CPU cores you need

### Option A: Cloud Run (simplest, but capped at 8 vCPUs per instance)

```bash
gcloud run deploy quantum-api \
  --image=$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/quantum-api:latest \
  --region=$REGION \
  --cpu=8 \
  --memory=8Gi \
  --timeout=900 \
  --allow-unauthenticated
```

**Important caveat for VQC specifically:** your server has 96 cores, and the
`ProcessPoolExecutor` in `app.py` sizes itself to whatever `os.cpu_count()`
reports at runtime — so on Cloud Run with 8 vCPUs, it'll use 8 workers
instead of 96. Rough math: if 96 cores got you to ~10 seconds, 8 cores would
land closer to ~2 minutes (not the ~10 minutes you started with, but not the
~10 seconds you're seeing now either). QSVC and QKNN are unaffected by this —
their speed comes from vectorized matrix math, not core count, so they'll
stay near-instant on Cloud Run regardless.

Also set `--timeout` generously (up to 3600s max on Cloud Run) since VQC
requests will take longer here than on your 96-core server.

### Option B: GKE or a Compute Engine VM (if you need VQC closer to your 96-core speed)

Cloud Run's 8 vCPU ceiling is a hard platform limit, not something you can
raise with quota requests. If VQC's ~2 minute (Option A) response time isn't
acceptable, run the same image on a Compute Engine VM or GKE node with more
cores instead:

```bash
# Example: deploy the same image to a GKE node pool with high-CPU machines,
# or run directly on a Compute Engine VM with Docker installed:
gcloud compute instances create-with-container quantum-api-vm \
  --container-image=$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/quantum-api:latest \
  --machine-type=c2-standard-60 \
  --zone=$REGION-a
```
(`c2-standard-60` gives 60 vCPUs — adjust the machine type to match how many
cores you actually want; GCE doesn't have Cloud Run's 8-vCPU cap.)

## 6. Verify env vars line up

If you don't bake `API_files/` into the image and mount it externally
instead (e.g. via a GCS FUSE volume on Cloud Run, or a persistent disk on
GCE), set `QAPI_FILES_DIR` to wherever it's mounted:

```bash
gcloud run deploy quantum-api \
  --image=... \
  --set-env-vars=QAPI_FILES_DIR=/mnt/api-files \
  ...
```
