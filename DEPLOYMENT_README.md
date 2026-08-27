# Qmodels API - Cloud Run deployment

This package is prepared for a Docker + Cloud Build + Cloud Run deployment.

Important:
- The application listens on `0.0.0.0` and honors Cloud Run's `PORT` environment variable.
- Cloud Build uses Artifact Registry in `us-central1`.
- Cloud Run service name: `qmodels-api`.
- The build is configured for 2 vCPU, 4 GiB memory, 900s request timeout, and concurrency 1.
- The existing model/data files are included.

Deployment choices:
1. Easiest UI route: connect the GitHub repo to Cloud Build and select `Dockerfile` as the build type.
2. YAML route: configure the trigger to use `cloudbuild.yaml`.
