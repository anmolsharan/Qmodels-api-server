# Qmodels API — Cloud Run Deployment

## Automatic deployment

This repository is configured for:

GitHub `main` -> Cloud Build trigger `qmodels-api-deploy` -> Docker -> Artifact Registry `qmodels` -> Cloud Run `qmodels-api`.

The Cloud Run service is configured by `cloudbuild.yaml` for:
- 4 vCPU
- 16 GiB RAM
- CPU boost
- 15 minute request timeout
- concurrency 1
- unauthenticated access

## Container paths

Runtime model/data files are under `/app/API_files`.
The application uses `QAPI_FILES_DIR` if supplied; otherwise it defaults to `/app/API_files`.

## FastAPI

The application listens on `0.0.0.0` and uses the Cloud Run `PORT` environment variable, defaulting to 8080.

Swagger UI:
`/docs`

OpenAPI:
`/openapi.json`

## Deployment

After changing code:

```cmd
git add .
git commit -m "Update Qmodels API"
git push
```

The Cloud Build trigger should deploy the new revision automatically.

## Important

Do not commit secrets, API keys, passwords, or private credentials.
