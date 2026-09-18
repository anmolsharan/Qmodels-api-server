# Qmodels API Server

FastAPI service for the QMine Quantum + Classical Models pipeline.

## Included endpoints

Existing endpoints retained:
- `POST /vqc/classification-metrics`
- `POST /qsvc/classification-metrics`
- `POST /qknn/classification-metrics`

Additional endpoints:
- `POST /vqc-amplitude/classification-metrics`
- `POST /vqc/probability-map`
- `POST /vqc-amplitude/probability-map`
- `POST /qsvc/probability-map`
- `POST /qknn/probability-map`
- `POST /rf/probability-map`
- `POST /rf/classification-metrics`
- `GET /health`
- `GET /`

## Repository layout

```text
Qmodels-api-server/
├── app.py
├── Dockerfile
├── cloudbuild.yaml
├── requirements.txt
├── .dockerignore
├── .gitignore
└── API_files/
    ├── artifacts_50/
    │   ├── amplitude_vqc_weights.pt
    │   ├── data_re-uploading_weights.pt
    │   ├── learnable_encoding_weights.pt
    │   ├── rx_encoding_weights.pt
    │   ├── ry_encoding_weights.pt
    │   └── rz_encoding_weights.pt
    ├── known_occurrences.csv
    ├── prediction_grid.csv
    ├── qknn_model.pkl
    ├── qsvc_38.pkl
    ├── random_forest_model.pkl
    ├── X_Test_QKNN.csv
    ├── X_Test_QSVC_38.csv
    ├── X_Test_RF.csv
    ├── X_Test_rfe_new.csv
    ├── X_train_norm_38.npy
    ├── y_Test_QKNN.csv
    ├── y_Test_QSVC_38.csv
    ├── y_Test_RF.csv
    └── y_Test_rfe_new.csv
```

All model/data paths are resolved from `QAPI_FILES_DIR`, which defaults to
`/app/API_files` inside the Docker container. The previous developer-specific
`/home/ankit/QAPIs/...` paths are not used.

## Cloud Run deployment

`cloudbuild.yaml` keeps the existing service name, region, port, CPU, memory,
timeout, and concurrency configuration:

- Service: `qmodels-api`
- Region: `us-central1`
- Port: `8080`
- CPU: `4`
- Memory: `16Gi`
- Timeout: `900s`
- Concurrency: `1`

Pushing to the configured GitHub/Cloud Build trigger can therefore rebuild and
redeploy the same Cloud Run service.

## Important

Do not delete the repository's existing `.git` directory when replacing the
working tree. The clean package in this ZIP intentionally does not include
`.git`; copy its contents into your existing local repository, replacing the
old files.
