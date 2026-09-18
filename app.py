# =============================================================================
# QMine Quantum + Classical Models FastAPI
# VQC + QSVC + QKNN + VQC-Amplitude + Random Forest
# =============================================================================

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pennylane as qml

import joblib
import pickle
import time
import logging
import os
import io
import multiprocessing as mp

from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

from qiskit import QuantumCircuit

from sklearn.preprocessing import normalize
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
)

from scipy.special import expit
from scipy.interpolate import griddata

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import uvicorn


# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("qmine-api")


# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(
    title="QMine Quantum + Classical Models API",
    description="API for VQC, QSVC, QKNN, VQC-Amplitude and Random Forest",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# REQUEST LOGGER
# =============================================================================

@app.middleware("http")
async def log_requests(request: Request, call_next):

    start_time = time.time()

    body = await request.body()

    logger.info(
        f"➡️ Incoming request: "
        f"{request.method} {request.url.path}"
    )

    if body:
        logger.info(
            f"📦 Payload: "
            f"{body.decode('utf-8', errors='ignore')}"
        )

    response = await call_next(request)

    process_time = (time.time() - start_time) * 1000

    logger.info(
        f"✅ Completed: "
        f"{request.method} {request.url.path} "
        f"| Status: {response.status_code} "
        f"| Time: {process_time:.2f} ms"
    )

    return response


# =============================================================================
# DEVICE
# =============================================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

logger.info(f"PyTorch device: {DEVICE}")


# =============================================================================
# BASE PATH
# =============================================================================

QAPI_FILES_DIR = Path(
    os.environ.get("QAPI_FILES_DIR", "/app/API_files")
)

VQC_ARTIFACT_DIR = QAPI_FILES_DIR / "artifacts_50"

logger.info(f"QAPI files directory: {QAPI_FILES_DIR}")
logger.info(f"VQC artifact directory: {VQC_ARTIFACT_DIR}")


# =============================================================================
# COMMON REQUEST MODELS
# =============================================================================

class MetricsRequest(BaseModel):

    model_type: Optional[str] = None

    features: List[str]


class ProbabilityMapRequest(BaseModel):

    target_name: str = "La"

    model_type: Optional[str] = None


# =============================================================================
# COMMON FUNCTIONS
# =============================================================================

def round_report(report: dict) -> dict:

    for key in report:

        if isinstance(report[key], dict):

            for metric in report[key]:

                if isinstance(
                    report[key][metric],
                    float
                ):
                    report[key][metric] = round(
                        report[key][metric],
                        2
                    )

        elif isinstance(
            report[key],
            float
        ):

            report[key] = round(
                report[key],
                2
            )

    return report


def extract_y_true(
    y_df: pd.DataFrame
) -> np.ndarray:

    if y_df.shape[1] == 1:
        return y_df.iloc[:, 0].values

    for col in [
        "label",
        "Label",
        "y",
        "target"
    ]:

        if col in y_df.columns:
            return y_df[col].values

    logger.warning(
        f"Could not find standard label column. "
        f"Using last column: {y_df.columns[-1]}"
    )

    return y_df.iloc[:, -1].values


def safe_normalize_rows(X):

    norms = np.linalg.norm(
        X,
        axis=1,
        keepdims=True
    )

    norms[norms == 0] = 1.0

    return X / norms


# =============================================================================
# ============================================================================
# 1. VQC ANGLE ENCODING
# ============================================================================
# =============================================================================

VQC_N_QUBITS = 20

VQC_N_LAYERS = 1

VQC_N_LAYERS_RU = 2

VQC_THRESHOLD = 0.5


VQC_VALID_FEATURES = [

    "fault_dist",
    "fold_dist",
    "pc1",
    "pc2",
    "pc3",

    "elevation1",

    "observed_1",
    "theoreti_1",
    "bouguer__1",

    "bg",
    "bg_residual",
    "bg_1vd",
    "bg_thd",

    "observed_m",
    "igrf_nt",
    "magnetic_a",

    "mag_tf_raw",
    "mag_tf",
    "mag_residual",
    "mag_as",
]


VQC_X_TEST_PATH = (
    QAPI_FILES_DIR / r"X_Test_rfe_new.csv"
)

VQC_Y_TEST_PATH = (
    QAPI_FILES_DIR / r"y_Test_rfe_new.csv"
)


# =============================================================================
# VQC DEVICE
# =============================================================================

VQC_BACKEND = (
    "lightning.gpu"
    if torch.cuda.is_available()
    else "lightning.qubit"
)

logger.info(
    f"VQC backend: {VQC_BACKEND}"
)

vqc_dev = qml.device(
    VQC_BACKEND,
    wires=VQC_N_QUBITS
)


# =============================================================================
# VQC ANSATZ
# =============================================================================

def vqc_ansatz(layer_w):

    for i in range(VQC_N_QUBITS):

        qml.RY(
            layer_w[i, 0],
            wires=i
        )

        qml.RZ(
            layer_w[i, 1],
            wires=i
        )

    for i in range(
        VQC_N_QUBITS - 1
    ):

        qml.CNOT(
            wires=[i, i + 1]
        )

    qml.CNOT(
        wires=[
            VQC_N_QUBITS - 1,
            0
        ]
    )


# =============================================================================
# VQC TORCH CIRCUITS
# =============================================================================

@qml.qnode(
    vqc_dev,
    interface="torch",
    diff_method="adjoint"
)
def vqc_circuit_rx(
    weights,
    x
):

    for i in range(VQC_N_QUBITS):

        qml.RX(
            x[i],
            wires=i
        )

    vqc_ansatz(
        weights[0]
    )

    return qml.expval(
        qml.PauliZ(0)
    )


@qml.qnode(
    vqc_dev,
    interface="torch",
    diff_method="adjoint"
)
def vqc_circuit_ry(
    weights,
    x
):

    for i in range(VQC_N_QUBITS):

        qml.RY(
            x[i],
            wires=i
        )

    vqc_ansatz(
        weights[0]
    )

    return qml.expval(
        qml.PauliZ(0)
    )


@qml.qnode(
    vqc_dev,
    interface="torch",
    diff_method="adjoint"
)
def vqc_circuit_rz(
    weights,
    x
):

    for i in range(VQC_N_QUBITS):

        qml.RZ(
            x[i],
            wires=i
        )

    vqc_ansatz(
        weights[0]
    )

    return qml.expval(
        qml.PauliZ(0)
    )


@qml.qnode(
    vqc_dev,
    interface="torch",
    diff_method="adjoint"
)
def vqc_circuit_learnable(
    weights,
    enc,
    x
):

    for i in range(VQC_N_QUBITS):

        qml.RY(
            enc[i, 0] * x[i]
            + enc[i, 1],
            wires=i
        )

    vqc_ansatz(
        weights[0]
    )

    return qml.expval(
        qml.PauliZ(0)
    )


@qml.qnode(
    vqc_dev,
    interface="torch",
    diff_method="adjoint"
)
def vqc_circuit_reupload(
    weights,
    x
):

    for r in range(
        VQC_N_LAYERS_RU
    ):

        for i in range(
            VQC_N_QUBITS
        ):

            qml.RY(
                x[i],
                wires=i
            )

        vqc_ansatz(
            weights[r]
        )

    return qml.expval(
        qml.PauliZ(0)
    )


# =============================================================================
# VQC NUMPY CIRCUITS
# =============================================================================

@qml.qnode(
    vqc_dev,
    interface="numpy"
)
def vqc_circuit_rx_np(
    weights,
    x
):

    for i in range(VQC_N_QUBITS):
        qml.RX(x[i], wires=i)

    vqc_ansatz(
        weights[0]
    )

    return qml.expval(
        qml.PauliZ(0)
    )


@qml.qnode(
    vqc_dev,
    interface="numpy"
)
def vqc_circuit_ry_np(
    weights,
    x
):

    for i in range(VQC_N_QUBITS):
        qml.RY(x[i], wires=i)

    vqc_ansatz(
        weights[0]
    )

    return qml.expval(
        qml.PauliZ(0)
    )


@qml.qnode(
    vqc_dev,
    interface="numpy"
)
def vqc_circuit_rz_np(
    weights,
    x
):

    for i in range(VQC_N_QUBITS):
        qml.RZ(x[i], wires=i)

    vqc_ansatz(
        weights[0]
    )

    return qml.expval(
        qml.PauliZ(0)
    )


@qml.qnode(
    vqc_dev,
    interface="numpy"
)
def vqc_circuit_learnable_np(
    weights,
    enc,
    x
):

    for i in range(VQC_N_QUBITS):

        qml.RY(
            enc[i, 0] * x[i]
            + enc[i, 1],
            wires=i
        )

    vqc_ansatz(
        weights[0]
    )

    return qml.expval(
        qml.PauliZ(0)
    )


@qml.qnode(
    vqc_dev,
    interface="numpy"
)
def vqc_circuit_reupload_np(
    weights,
    x
):

    for r in range(
        VQC_N_LAYERS_RU
    ):

        for i in range(
            VQC_N_QUBITS
        ):

            qml.RY(
                x[i],
                wires=i
            )

        vqc_ansatz(
            weights[r]
        )

    return qml.expval(
        qml.PauliZ(0)
    )


# =============================================================================
# VQC PARALLEL PROCESSING
# =============================================================================

_VQC_POOL = None

_VQC_MAX_WORKERS = max(
    1,
    os.cpu_count() or 1
)


def _get_vqc_pool():

    global _VQC_POOL

    if _VQC_POOL is None:

        ctx = mp.get_context(
            "fork"
        )

        _VQC_POOL = ProcessPoolExecutor(
            max_workers=_VQC_MAX_WORKERS,
            mp_context=ctx
        )

        logger.info(
            f"[VQC] Started "
            f"{_VQC_MAX_WORKERS} workers"
        )

    return _VQC_POOL


def _eval_angle_chunk(
    circuit_key,
    weights_np,
    x_chunk
):

    circuit_fn = {

        "rx":
            vqc_circuit_rx_np,

        "ry":
            vqc_circuit_ry_np,

        "rz":
            vqc_circuit_rz_np,

    }[circuit_key]

    return np.array(
        [
            circuit_fn(
                weights_np,
                x
            )
            for x in x_chunk
        ]
    )


def _eval_learnable_chunk(
    weights_np,
    enc_np,
    x_chunk
):

    return np.array(
        [
            vqc_circuit_learnable_np(
                weights_np,
                enc_np,
                x
            )
            for x in x_chunk
        ]
    )


def _eval_reupload_chunk(
    weights_np,
    x_chunk
):

    return np.array(
        [
            vqc_circuit_reupload_np(
                weights_np,
                x
            )
            for x in x_chunk
        ]
    )


def _parallel_eval(
    worker_fn,
    fixed_args,
    X_batch
):

    if len(X_batch) == 0:
        return np.array([])

    pool = _get_vqc_pool()

    n_workers = min(
        _VQC_MAX_WORKERS,
        len(X_batch)
    )

    chunks = np.array_split(
        X_batch,
        n_workers
    )

    chunks = [
        c for c in chunks
        if len(c) > 0
    ]

    futures = [

        pool.submit(
            worker_fn,
            *fixed_args,
            chunk
        )

        for chunk in chunks
    ]

    results = [
        f.result()
        for f in futures
    ]

    return np.concatenate(
        results
    )


# =============================================================================
# VQC MODEL CLASSES
# =============================================================================

class AngleVQC(nn.Module):

    def __init__(
        self,
        circuit,
        n_layers=VQC_N_LAYERS,
        circuit_key=None
    ):

        super().__init__()

        self.circuit = circuit

        self.circuit_key = circuit_key

        self.weights = nn.Parameter(

            torch.tensor(

                np.random.uniform(
                    -np.pi,
                    np.pi,
                    (
                        n_layers,
                        VQC_N_QUBITS,
                        2
                    )
                ),

                dtype=torch.float32
            )
        )

    def forward(
        self,
        X_batch
    ):

        weights_np = (
            self.weights
            .detach()
            .cpu()
            .numpy()
        )

        X_np = (
            X_batch
            .detach()
            .cpu()
            .numpy()
        )

        out_np = _parallel_eval(

            _eval_angle_chunk,

            (
                self.circuit_key,
                weights_np
            ),

            X_np
        )

        out = torch.tensor(
            out_np,
            dtype=torch.float32,
            device=X_batch.device
        )

        return (
            out + 1.0
        ) / 2.0


class LearnableVQC(nn.Module):

    def __init__(
        self,
        n_layers=VQC_N_LAYERS
    ):

        super().__init__()

        self.weights = nn.Parameter(

            torch.tensor(

                np.random.uniform(
                    -np.pi,
                    np.pi,
                    (
                        n_layers,
                        VQC_N_QUBITS,
                        2
                    )
                ),

                dtype=torch.float32
            )
        )

        enc = torch.zeros(
            VQC_N_QUBITS,
            2,
            dtype=torch.float32
        )

        enc[:, 0] = 1.0

        self.enc = nn.Parameter(
            enc
        )

    def forward(
        self,
        X_batch
    ):

        weights_np = (
            self.weights
            .detach()
            .cpu()
            .numpy()
        )

        enc_np = (
            self.enc
            .detach()
            .cpu()
            .numpy()
        )

        X_np = (
            X_batch
            .detach()
            .cpu()
            .numpy()
        )

        out_np = _parallel_eval(

            _eval_learnable_chunk,

            (
                weights_np,
                enc_np
            ),

            X_np
        )

        out = torch.tensor(
            out_np,
            dtype=torch.float32,
            device=X_batch.device
        )

        return (
            out + 1.0
        ) / 2.0


class ReuploadVQC(nn.Module):

    def __init__(
        self,
        n_layers=VQC_N_LAYERS_RU
    ):

        super().__init__()

        self.weights = nn.Parameter(

            torch.tensor(

                np.random.uniform(
                    -np.pi,
                    np.pi,
                    (
                        n_layers,
                        VQC_N_QUBITS,
                        2
                    )
                ),

                dtype=torch.float32
            )
        )

    def forward(
        self,
        X_batch
    ):

        weights_np = (
            self.weights
            .detach()
            .cpu()
            .numpy()
        )

        X_np = (
            X_batch
            .detach()
            .cpu()
            .numpy()
        )

        out_np = _parallel_eval(

            _eval_reupload_chunk,

            (
                weights_np,
            ),

            X_np
        )

        out = torch.tensor(
            out_np,
            dtype=torch.float32,
            device=X_batch.device
        )

        return (
            out + 1.0
        ) / 2.0


def vqc_load_model(
    model_obj,
    path
):

    path = Path(path)

    if not path.exists():

        raise FileNotFoundError(
            f"VQC weights not found: {path}"
        )

    model_obj.load_state_dict(

        torch.load(
            path,
            map_location=DEVICE,
            weights_only=True
        )
    )

    model_obj.eval()

    return model_obj.to(
        DEVICE
    )


# =============================================================================
# LOAD VQC MODELS
# =============================================================================

vqc_model_rx = vqc_load_model(

    AngleVQC(
        vqc_circuit_rx,
        circuit_key="rx"
    ),

    VQC_ARTIFACT_DIR / r"rx_encoding_weights.pt"
)


vqc_model_ry = vqc_load_model(

    AngleVQC(
        vqc_circuit_ry,
        circuit_key="ry"
    ),

    VQC_ARTIFACT_DIR / r"ry_encoding_weights.pt"
)


vqc_model_rz = vqc_load_model(

    AngleVQC(
        vqc_circuit_rz,
        circuit_key="rz"
    ),

    VQC_ARTIFACT_DIR / r"rz_encoding_weights.pt"
)


vqc_model_learnable = vqc_load_model(

    LearnableVQC(),

    VQC_ARTIFACT_DIR / r"learnable_encoding_weights.pt"
)


vqc_model_reupload = vqc_load_model(

    ReuploadVQC(),

    VQC_ARTIFACT_DIR / r"data_re-uploading_weights.pt"
)


VQC_MODEL_MAPPING = {

    "rx": {
        "model": vqc_model_rx,
        "name": "RX Encoding"
    },

    "ry": {
        "model": vqc_model_ry,
        "name": "RY Encoding"
    },

    "rz": {
        "model": vqc_model_rz,
        "name": "RZ Encoding"
    },

    "learnable": {
        "model": vqc_model_learnable,
        "name": "Learnable Encoding"
    },

    "data_reuploading": {
        "model": vqc_model_reupload,
        "name": "Data Re-uploading"
    },
}


# =============================================================================
# VQC CLASSIFICATION
# =============================================================================

@app.post(
    "/vqc/classification-metrics"
)
def vqc_classification_metrics(
    req: MetricsRequest
):

    logger.info(
        f"[VQC] model={req.model_type}"
    )

    if req.model_type not in VQC_MODEL_MAPPING:

        raise HTTPException(

            status_code=400,

            detail=(
                "Invalid model_type. "
                f"Use: "
                f"{list(VQC_MODEL_MAPPING.keys())}"
            )
        )

    try:

        X_test = pd.read_csv(
            VQC_X_TEST_PATH
        )

        y_test = pd.read_csv(
            VQC_Y_TEST_PATH
        )

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    valid_requested = [
        f for f in req.features
        if f in X_test.columns
    ]

    ignored_features = [
        f for f in req.features
        if f not in X_test.columns
    ]

    if len(valid_requested) != VQC_N_QUBITS:

        raise HTTPException(

            status_code=400,

            detail=(
                f"Need exactly "
                f"{VQC_N_QUBITS} valid features. "
                f"Got {len(valid_requested)}."
            )
        )

    X_selected = X_test[
        valid_requested
    ]

    X_tensor = torch.tensor(
        X_selected.values,
        dtype=torch.float32
    )

    selected_model = (
        VQC_MODEL_MAPPING[
            req.model_type
        ]["model"]
    )

    model_name = (
        VQC_MODEL_MAPPING[
            req.model_type
        ]["name"]
    )

    with torch.no_grad():

        probabilities = (
            selected_model(
                X_tensor.to(DEVICE)
            )
            .cpu()
            .numpy()
            .flatten()
        )

    predictions = (
        probabilities >= VQC_THRESHOLD
    ).astype(int)

    y_true = extract_y_true(
        y_test
    )

    if len(y_true) != len(predictions):

        raise HTTPException(

            status_code=400,

            detail=(
                f"y_test has {len(y_true)} rows "
                f"but predictions have "
                f"{len(predictions)} rows."
            )
        )

    accuracy = accuracy_score(
        y_true,
        predictions
    )

    auc_score = roc_auc_score(
        y_true,
        probabilities
    )

    report = round_report(

        classification_report(
            y_true,
            predictions,
            output_dict=True,
            zero_division=0
        )
    )

    return {

        "model_used":
            model_name,

        "classification_accuracy":
            round(float(accuracy), 4),

        "auc_roc":
            round(float(auc_score), 4),

        "classification_report":
            report,

        "ignored_features":
            ignored_features,
    }


# =============================================================================
# ============================================================================
# 2. QSVC
# ============================================================================
# =============================================================================

QSVC_N_QUBITS = 6

QSVC_AMP_DIM = 64

QSVC_N_FEATURES = 44


QSVC_X_TEST_PATH = (
    QAPI_FILES_DIR / r"X_Test_QSVC_38.csv"
)

QSVC_Y_TEST_PATH = (
    QAPI_FILES_DIR / r"y_Test_QSVC_38.csv"
)

QSVC_XTRAIN_NORM_PATH = (
    QAPI_FILES_DIR / r"X_train_norm_38.npy"
)

QSVC_MODEL_PATH = (
    QAPI_FILES_DIR / r"qsvc_38.pkl"
)


QSVC_VALID_FEATURES = [

    "fault_dist",
    "fold_dist",
    "PC1",
    "PC2",
    "PC3",

    "elevation1",

    "observed_1",
    "theoreti_1",
    "bouguer__1",

    "BG",
    "BG_Residual",
    "BG_1VD",
    "BG_THD",
    "BG_TDR",

    "observed_m",
    "igrf_nt",
    "magnetic_a",

    "MAG_TF_raw",
    "MAG_TF",
    "MAG_Residual",
    "MAG_1VZ",
    "MAG_AS",

    "Amphibolite–Metabasic rock",
    "Brecciated ferruginous quartzite",
    "Calc-gneiss",
    "Carbonaceous phyllite with schist",
    "Conglomerate",
    "Dolomitic marble",
    "Fine sand, silt and clay",
    "Gneiss",
    "Granite",
    "Granite gneiss",
    "Graphite schist",
    "Meta-dolerite",
    "Other",
    "Pegmatite",
    "Phyllite",
    "Quartz vein",
    "Quartzite",
    "Quartzite with Phyllite",
    "Schist",
    "Calc-silicate",
    "Carbonate rock",
    "Migmatite",
]


def qsvc_normalize_amplitude(
    x
):

    norm = np.linalg.norm(x)

    if norm > 0:
        return x / norm

    return x


def qsvc_compute_kernel(
    X1,
    X2
):

    return (
        X1 @ X2.T
    ) ** 2


# =============================================================================
# LOAD QSVC
# =============================================================================

qsvc_model = joblib.load(
    QSVC_MODEL_PATH
)

qsvc_X_train_norm = np.load(
    QSVC_XTRAIN_NORM_PATH
)

logger.info(
    "[QSVC] model loaded"
)


# =============================================================================
# QSVC CLASSIFICATION
# =============================================================================

@app.post(
    "/qsvc/classification-metrics"
)
def qsvc_classification_metrics(
    req: MetricsRequest
):

    try:

        X_test = pd.read_csv(
            QSVC_X_TEST_PATH
        )

        y_test = pd.read_csv(
            QSVC_Y_TEST_PATH
        )

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    valid_requested = [
        f for f in req.features
        if f in X_test.columns
    ]

    ignored_features = [
        f for f in req.features
        if f not in X_test.columns
    ]

    if len(valid_requested) != QSVC_N_FEATURES:

        raise HTTPException(

            status_code=400,

            detail=(
                f"Need exactly "
                f"{QSVC_N_FEATURES} valid features. "
                f"Got {len(valid_requested)}."
            )
        )

    X_selected = (
        X_test[
            valid_requested
        ].values
    )

    X_selected_norm = np.array(

        [
            qsvc_normalize_amplitude(x)
            for x in X_selected
        ]
    )

    K_test = qsvc_compute_kernel(
        X_selected_norm,
        qsvc_X_train_norm
    )

    y_pred = qsvc_model.predict(
        K_test
    )

    y_score = (
        qsvc_model
        .decision_function(K_test)
    )

    y_true = extract_y_true(
        y_test
    )

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    auc_score = roc_auc_score(
        y_true,
        y_score
    )

    report = round_report(

        classification_report(
            y_true,
            y_pred,
            output_dict=True,
            zero_division=0
        )
    )

    return {

        "model_used":
            "QSVC (Amplitude-Encoded Kernel)",

        "classification_accuracy":
            round(float(accuracy), 4),

        "auc_roc":
            round(float(auc_score), 4),

        "classification_report":
            report,

        "ignored_features":
            ignored_features,
    }


# =============================================================================
# ============================================================================
# 3. QKNN
# ============================================================================
# =============================================================================

QKNN_N_FEATURES = 44

QKNN_N_QUBITS = 6

QKNN_AMP_DIM = 64


QKNN_X_TEST_PATH = (
    QAPI_FILES_DIR /
    "X_Test_QKNN.csv"
)

QKNN_Y_TEST_PATH = (
    QAPI_FILES_DIR /
    "y_Test_QKNN.csv"
)

QKNN_MODEL_PATH = (
    QAPI_FILES_DIR /
    "qknn_model.pkl"
)


QKNN_VALID_FEATURES = QSVC_VALID_FEATURES


qknn_dev = qml.device(
    "lightning.qubit",
    wires=QKNN_N_QUBITS
)


@qml.qnode(qknn_dev)
def qknn_fidelity_circuit(
    x1,
    x2
):

    qml.StatePrep(
        x1,
        wires=range(QKNN_N_QUBITS)
    )

    qml.adjoint(
        qml.StatePrep
    )(
        x2,
        wires=range(QKNN_N_QUBITS)
    )

    return qml.probs(
        wires=range(QKNN_N_QUBITS)
    )


class QuantumKNN:

    def __init__(
        self,
        k=5
    ):

        self.k = k

        self.X_train = None

        self.y_train = None


    def fit(
        self,
        X,
        y
    ):

        self.X_train = X

        self.y_train = y


    def _compute_distances(
        self,
        x
    ):

        fidelities = (
            self.X_train @ x
        ) ** 2

        return (
            1.0 -
            fidelities
        )


    def predict_single(
        self,
        x
    ):

        distances = (
            self._compute_distances(x)
        )

        knn_idx = np.argsort(
            distances
        )[:self.k]

        knn_labels = (
            self.y_train[
                knn_idx
            ]
        )

        return Counter(
            knn_labels
        ).most_common(1)[0][0]


    def predict(
        self,
        X
    ):

        return np.array(

            [
                self.predict_single(x)
                for x in X
            ]
        )


    def predict_proba(
        self,
        X
    ):

        probas = []

        for x in X:

            distances = (
                self._compute_distances(x)
            )

            knn_idx = np.argsort(
                distances
            )[:self.k]

            knn_dists = (
                distances[
                    knn_idx
                ] + 1e-10
            )

            weights = (
                1.0 /
                knn_dists
            )

            weights /= weights.sum()

            prob1 = sum(

                w

                for w, lbl

                in zip(
                    weights,
                    self.y_train[
                        knn_idx
                    ]
                )

                if lbl == 1
            )

            probas.append(
                [
                    1 - prob1,
                    prob1
                ]
            )

        return np.array(
            probas
        )


def qknn_load_model(
    path
):

    with open(
        path,
        "rb"
    ) as f:

        state = pickle.load(f)

    model = QuantumKNN(
        k=state["k"]
    )

    model.X_train = (
        state["X_train"]
    )

    model.y_train = (
        state["y_train"]
    )

    return model


qknn_model = qknn_load_model(
    QKNN_MODEL_PATH
)

logger.info(
    "[QKNN] model loaded"
)


# =============================================================================
# QKNN CLASSIFICATION
# =============================================================================

@app.post(
    "/qknn/classification-metrics"
)
def qknn_classification_metrics(
    req: MetricsRequest
):

    try:

        X_test = pd.read_csv(
            QKNN_X_TEST_PATH
        )

        y_test = pd.read_csv(
            QKNN_Y_TEST_PATH
        )

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    valid_requested = [
        f for f in req.features
        if f in X_test.columns
    ]

    ignored_features = [
        f for f in req.features
        if f not in X_test.columns
    ]

    if len(valid_requested) != QKNN_N_FEATURES:

        raise HTTPException(

            status_code=400,

            detail=(
                f"Need exactly "
                f"{QKNN_N_FEATURES} valid features. "
                f"Got {len(valid_requested)}."
            )
        )

    X_selected = (
        X_test[
            valid_requested
        ].values
    )

    X_pad = np.pad(

        X_selected,

        (
            (0, 0),
            (
                0,
                QKNN_AMP_DIM -
                QKNN_N_FEATURES
            )
        )
    )

    X_norm = normalize(
        X_pad,
        norm="l2"
    )

    y_pred = qknn_model.predict(
        X_norm
    )

    y_score = (
        qknn_model
        .predict_proba(X_norm)[:, 1]
    )

    y_true = extract_y_true(
        y_test
    )

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    auc_score = roc_auc_score(
        y_true,
        y_score
    )

    report = round_report(

        classification_report(
            y_true,
            y_pred,
            output_dict=True,
            zero_division=0
        )
    )

    return {

        "model_used":
            f"QKNN (k={qknn_model.k}, Fidelity Distance)",

        "classification_accuracy":
            round(float(accuracy), 4),

        "auc_roc":
            round(float(auc_score), 4),

        "classification_report":
            report,

        "ignored_features":
            ignored_features,
    }


# =============================================================================
# ============================================================================
# 4. VQC AMPLITUDE ENCODING
# ============================================================================
# =============================================================================

VQC_AMP_N_QUBITS = 6

VQC_AMP_N_LAYERS = 3

VQC_AMP_N_FEATURES = 44

VQC_AMP_DIM = 64


VQC_AMP_X_TEST_PATH = (
    QKNN_X_TEST_PATH
)

VQC_AMP_Y_TEST_PATH = (
    QKNN_Y_TEST_PATH
)

VQC_AMP_VALID_FEATURES = (
    QKNN_VALID_FEATURES
)


VQC_AMPLITUDE_WEIGHTS_PATH = (
    VQC_ARTIFACT_DIR /
    "amplitude_vqc_weights.pt"
)


vqc_amp_dev = qml.device(
    VQC_BACKEND,
    wires=VQC_AMP_N_QUBITS
)


@qml.qnode(
    vqc_amp_dev,
    interface="torch",
    diff_method="adjoint"
)
def vqc_amplitude_circuit(
    weights,
    x
):

    qml.StatePrep(
        x,
        wires=range(
            VQC_AMP_N_QUBITS
        )
    )

    for l in range(
        VQC_AMP_N_LAYERS
    ):

        for i in range(
            VQC_AMP_N_QUBITS
        ):

            qml.RY(
                weights[l, i, 0],
                wires=i
            )

            qml.RZ(
                weights[l, i, 1],
                wires=i
            )

        for i in range(
            VQC_AMP_N_QUBITS - 1
        ):

            qml.CNOT(
                wires=[i, i + 1]
            )

        qml.CNOT(
            wires=[
                VQC_AMP_N_QUBITS - 1,
                0
            ]
        )

    return qml.expval(
        qml.PauliZ(0)
    )


@qml.qnode(
    vqc_amp_dev,
    interface="numpy"
)
def vqc_amplitude_circuit_np(
    weights,
    x
):

    qml.StatePrep(
        x,
        wires=range(
            VQC_AMP_N_QUBITS
        )
    )

    for l in range(
        VQC_AMP_N_LAYERS
    ):

        for i in range(
            VQC_AMP_N_QUBITS
        ):

            qml.RY(
                weights[l, i, 0],
                wires=i
            )

            qml.RZ(
                weights[l, i, 1],
                wires=i
            )

        for i in range(
            VQC_AMP_N_QUBITS - 1
        ):

            qml.CNOT(
                wires=[i, i + 1]
            )

        qml.CNOT(
            wires=[
                VQC_AMP_N_QUBITS - 1,
                0
            ]
        )

    return qml.expval(
        qml.PauliZ(0)
    )


def _eval_amplitude_vqc_chunk(
    weights_np,
    x_chunk
):

    return np.array(

        [
            vqc_amplitude_circuit_np(
                weights_np,
                x
            )

            for x in x_chunk
        ]
    )


class AmplitudeVQC(nn.Module):

    def __init__(
        self,
        n_layers=VQC_AMP_N_LAYERS
    ):

        super().__init__()

        self.weights = nn.Parameter(

            torch.tensor(

                np.random.uniform(
                    -np.pi,
                    np.pi,
                    (
                        n_layers,
                        VQC_AMP_N_QUBITS,
                        2
                    )
                ),

                dtype=torch.float32
            )
        )


    def forward(
        self,
        X_batch
    ):

        weights_np = (
            self.weights
            .detach()
            .cpu()
            .numpy()
        )

        X_np = (
            X_batch
            .detach()
            .cpu()
            .numpy()
        )

        out_np = _parallel_eval(

            _eval_amplitude_vqc_chunk,

            (
                weights_np,
            ),

            X_np
        )

        out = torch.tensor(
            out_np,
            dtype=torch.float32,
            device=X_batch.device
        )

        return (
            out + 1.0
        ) / 2.0


vqc_model_amplitude = None


try:

    vqc_model_amplitude = (
        vqc_load_model(
            AmplitudeVQC(),
            VQC_AMPLITUDE_WEIGHTS_PATH
        )
    )

    logger.info(
        "[VQC-Amplitude] weights loaded"
    )

except Exception as e:

    logger.warning(
        f"[VQC-Amplitude] "
        f"weights not loaded: {e}"
    )


# =============================================================================
# VQC AMPLITUDE CLASSIFICATION
# =============================================================================

@app.post(
    "/vqc-amplitude/classification-metrics"
)
def vqc_amplitude_classification_metrics(
    req: MetricsRequest
):

    if vqc_model_amplitude is None:

        raise HTTPException(

            status_code=503,

            detail=(
                "Amplitude VQC weights are "
                f"not available at: "
                f"{VQC_AMPLITUDE_WEIGHTS_PATH}"
            )
        )

    try:

        X_test = pd.read_csv(
            VQC_AMP_X_TEST_PATH
        )

        y_test = pd.read_csv(
            VQC_AMP_Y_TEST_PATH
        )

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    valid_requested = [
        f for f in req.features
        if f in X_test.columns
    ]

    ignored_features = [
        f for f in req.features
        if f not in X_test.columns
    ]

    if len(valid_requested) != 44:

        raise HTTPException(

            status_code=400,

            detail=(
                "Need exactly 44 valid features. "
                f"Got {len(valid_requested)}."
            )
        )

    X_selected = (
        X_test[
            valid_requested
        ].values
    )

    X_pad = np.pad(

        X_selected,

        (
            (0, 0),
            (0, 20)
        )
    )

    X_norm = normalize(
        X_pad,
        norm="l2"
    )

    X_tensor = torch.tensor(
        X_norm,
        dtype=torch.float32
    )

    with torch.no_grad():

        probabilities = (

            vqc_model_amplitude(
                X_tensor.to(DEVICE)
            )

            .cpu()
            .numpy()
            .flatten()
        )

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    y_true = extract_y_true(
        y_test
    )

    accuracy = accuracy_score(
        y_true,
        predictions
    )

    auc_score = roc_auc_score(
        y_true,
        probabilities
    )

    report = round_report(

        classification_report(
            y_true,
            predictions,
            output_dict=True,
            zero_division=0
        )
    )

    return {

        "model_used":
            "VQC (Amplitude Encoding, StatePrep + RY/RZ + CNOT)",

        "classification_accuracy":
            round(float(accuracy), 4),

        "auc_roc":
            round(float(auc_score), 4),

        "classification_report":
            report,

        "ignored_features":
            ignored_features,
    }


# =============================================================================
# ============================================================================
# 5. PROBABILITY MAP DATA
# ============================================================================
# =============================================================================

# IMPORTANT:
# The original code had:
#
# Local artifact path: API_files/prediction_grid.csv
#
# which was WRONG.
#
# Correct paths are:

GRID_CSV_PATH = (
    QAPI_FILES_DIR / r"prediction_grid.csv"
)

KNOWN_OCCURRENCE_PATH = (
    QAPI_FILES_DIR / r"known_occurrences.csv"
)


_grid_df_cache = None

_known_df_cache = None


def _load_grid_and_known():

    global _grid_df_cache, _known_df_cache

    if _grid_df_cache is None:

        if not GRID_CSV_PATH.exists():

            raise FileNotFoundError(
                f"Prediction grid not found: "
                f"{GRID_CSV_PATH}"
            )

        _grid_df_cache = pd.read_csv(
            GRID_CSV_PATH
        )

        logger.info(
            f"[Maps] Grid loaded: "
            f"{_grid_df_cache.shape}"
        )

    if _known_df_cache is None:

        if not KNOWN_OCCURRENCE_PATH.exists():

            raise FileNotFoundError(
                f"Known occurrences not found: "
                f"{KNOWN_OCCURRENCE_PATH}"
            )

        _known_df_cache = pd.read_csv(
            KNOWN_OCCURRENCE_PATH
        )

        logger.info(
            f"[Maps] Known occurrences loaded: "
            f"{_known_df_cache.shape}"
        )

    return (
        _grid_df_cache,
        _known_df_cache
    )


# =============================================================================
# MAP RENDERER
# =============================================================================

def render_probability_map_png(
    grid_df,
    probabilities,
    known_occurrence_df,
    model_name,
    target_name,
    x_col="X",
    y_col="Y",
    known_label_col="Label",
    figsize=(6, 9)
):

    if x_col not in grid_df.columns:

        raise HTTPException(
            status_code=400,
            detail=(
                f"Grid is missing "
                f"coordinate column: {x_col}"
            )
        )

    if y_col not in grid_df.columns:

        raise HTTPException(
            status_code=400,
            detail=(
                f"Grid is missing "
                f"coordinate column: {y_col}"
            )
        )

    x = grid_df[
        x_col
    ].values

    y = grid_df[
        y_col
    ].values

    x_unique = np.sort(
        grid_df[
            x_col
        ].unique()
    )

    y_unique = np.sort(
        grid_df[
            y_col
        ].unique()
    )

    is_regular_grid = (

        len(x_unique) *
        len(y_unique)

        ==

        len(grid_df)
    )

    fig, ax = plt.subplots(
        figsize=figsize
    )

    if is_regular_grid:

        temp_df = grid_df.copy()

        temp_df["prob"] = (
            probabilities
        )

        prob_grid = (

            temp_df

            .pivot(
                index=y_col,
                columns=x_col,
                values="prob"
            )

            .sort_index(
                ascending=True
            )
        )

        extent = [

            x_unique.min(),
            x_unique.max(),

            y_unique.min(),
            y_unique.max(),
        ]

        im = ax.imshow(

            prob_grid.values,

            extent=extent,

            origin="lower",

            aspect="auto",

            cmap="RdYlBu_r",

            vmin=0,

            vmax=1
        )

    else:

        xi = np.linspace(
            x.min(),
            x.max(),
            400
        )

        yi = np.linspace(
            y.min(),
            y.max(),
            700
        )

        Xi, Yi = np.meshgrid(
            xi,
            yi
        )

        Zi = griddata(

            (
                x,
                y
            ),

            probabilities,

            (
                Xi,
                Yi
            ),

            method="linear"
        )

        im = ax.imshow(

            Zi,

            extent=[
                x.min(),
                x.max(),
                y.min(),
                y.max()
            ],

            origin="lower",

            aspect="auto",

            cmap="RdYlBu_r",

            vmin=0,

            vmax=1
        )

    ax.set_title(
        f"{model_name} Probability Map — "
        f"{target_name}"
    )

    ax.set_xlabel(
        "Longitude"
    )

    ax.set_ylabel(
        "Latitude"
    )

    cbar = fig.colorbar(
        im,
        ax=ax
    )

    cbar.set_label(
        f"Predicted probability of "
        f"{target_name} occurrence"
    )

    fig.text(

        0.5,
        -0.02,

        f"Predicted {target_name} probability",

        ha="center",

        style="italic",

        fontsize=10
    )

    plt.tight_layout()

    buffer = io.BytesIO()

    fig.savefig(

        buffer,

        format="png",

        dpi=150,

        bbox_inches="tight"
    )

    plt.close(fig)

    buffer.seek(0)

    return buffer.getvalue()


# =============================================================================
# ============================================================================
# 6. QSVC PROBABILITY MAP
# ============================================================================
# =============================================================================

@app.post(
    "/qsvc/probability-map"
)
def qsvc_probability_map(
    req: ProbabilityMapRequest
):

    logger.info(
        f"[QSVC-Map] "
        f"target={req.target_name}"
    )

    try:

        grid_df, known_df = (
            _load_grid_and_known()
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    missing = [

        f

        for f in QSVC_VALID_FEATURES

        if f not in grid_df.columns
    ]

    if missing:

        raise HTTPException(

            status_code=400,

            detail=(
                "Grid CSV is missing columns: "
                f"{missing}"
            )
        )

    grid_X = (
        grid_df[
            QSVC_VALID_FEATURES
        ].values
    )

    grid_X_norm = (
        safe_normalize_rows(
            grid_X
        )
    )

    K_grid = (
        qsvc_compute_kernel(
            grid_X_norm,
            qsvc_X_train_norm
        )
    )

    decision_scores = (
        qsvc_model
        .decision_function(K_grid)
    )

    probs = expit(
        decision_scores
    )

    png_bytes = (
        render_probability_map_png(
            grid_df,
            probs,
            known_df,
            "QSVC",
            req.target_name
        )
    )

    # -----------------------------------------
    # Convert PNG bytes -> Base64
    # -----------------------------------------

    image_base64 = base64.b64encode(
        png_bytes
    ).decode("utf-8")

    # -----------------------------------------
    # Return JSON response
    # -----------------------------------------

    return JSONResponse(
        content={
            "target_name": req.target_name,
            "model": "QSVC",
            "image": image_base64
        }
    )


# =============================================================================
# ============================================================================
# 7. QKNN PROBABILITY MAP
# ============================================================================
# =============================================================================

@app.post(
    "/qknn/probability-map"
)
def qknn_probability_map(
    req: ProbabilityMapRequest
):

    logger.info(
        f"[QKNN-Map] "
        f"target={req.target_name}"
    )

    try:

        grid_df, known_df = (
            _load_grid_and_known()
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    missing = [

        f

        for f in QKNN_VALID_FEATURES

        if f not in grid_df.columns
    ]

    if missing:

        raise HTTPException(

            status_code=400,

            detail=(
                "Grid CSV is missing columns: "
                f"{missing}"
            )
        )

    grid_X = (
        grid_df[
            QKNN_VALID_FEATURES
        ].values
    )

    grid_X_pad = np.pad(

        grid_X,

        (
            (0, 0),
            (0, 20)
        )
    )

    grid_X_norm = normalize(
        grid_X_pad,
        norm="l2"
    )

    probs = (
        qknn_model
        .predict_proba(
            grid_X_norm
        )[:, 1]
    )

    png_bytes = (
        render_probability_map_png(

            grid_df,

            probs,

            known_df,

            f"QKNN (k={qknn_model.k})",

            req.target_name
        )
    )

    # -----------------------------------------
    # Convert PNG bytes -> Base64
    # -----------------------------------------

    image_base64 = base64.b64encode(
        png_bytes
    ).decode("utf-8")

    # -----------------------------------------
    # Return JSON response
    # -----------------------------------------

    return JSONResponse(
        content={
            "target_name": req.target_name,
            "model": f"QKNN (k={qknn_model.k})",
            "image": image_base64
        }
    )


# =============================================================================
# ============================================================================
# 8. VQC PROBABILITY MAP
# ============================================================================
# =============================================================================
@app.post("/vqc/probability-map")
def vqc_probability_map(
    req: ProbabilityMapRequest
):

    logger.info(
        f"[VQC-Map] model={req.model_type} "
        f"target={req.target_name}"
    )

    # ---------------------------------------------------------
    # Check model type
    # ---------------------------------------------------------

    if req.model_type not in VQC_MODEL_MAPPING:

        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid model_type. "
                f"Use one of: "
                f"{list(VQC_MODEL_MAPPING.keys())}"
            )
        )

    # ---------------------------------------------------------
    # Load grid and known occurrences
    # ---------------------------------------------------------

    try:

        grid_df, known_df = _load_grid_and_known()

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    # ---------------------------------------------------------
    # Resolve VQC feature names case-insensitively
    # ---------------------------------------------------------

    actual_columns = list(grid_df.columns)

    # Map lowercase column name -> actual CSV column name
    column_lookup = {
        str(col).strip().lower(): col
        for col in actual_columns
    }

    resolved_features = []
    missing_features = []

    for feature in VQC_VALID_FEATURES:

        key = feature.strip().lower()

        if key in column_lookup:

            resolved_features.append(
                column_lookup[key]
            )

        else:

            missing_features.append(
                feature
            )

    # ---------------------------------------------------------
    # Stop if any feature is actually missing
    # ---------------------------------------------------------

    if missing_features:

        raise HTTPException(

            status_code=400,

            detail={
                "message":
                    "Grid CSV is missing VQC features",

                "missing_features":
                    missing_features,

                "available_columns":
                    actual_columns
            }
        )

    logger.info(
        f"[VQC-Map] Resolved features: "
        f"{resolved_features}"
    )

    # ---------------------------------------------------------
    # Extract features in EXACT VQC order
    # ---------------------------------------------------------

    grid_X = (
        grid_df[
            resolved_features
        ].values
    )

    # ---------------------------------------------------------
    # Convert to float32
    # ---------------------------------------------------------

    try:

        grid_X = grid_X.astype(
            np.float32
        )

    except Exception as e:

        raise HTTPException(

            status_code=400,

            detail=(
                "VQC grid features could not "
                f"be converted to numeric values: {e}"
            )
        )

    # ---------------------------------------------------------
    # Check dimensions
    # ---------------------------------------------------------

    if grid_X.shape[1] != VQC_N_QUBITS:

        raise HTTPException(

            status_code=400,

            detail=(
                f"VQC requires "
                f"{VQC_N_QUBITS} features, "
                f"but received "
                f"{grid_X.shape[1]}"
            )
        )

    # ---------------------------------------------------------
    # Convert to Torch tensor
    # ---------------------------------------------------------

    grid_tensor = torch.tensor(
        grid_X,
        dtype=torch.float32
    )

    # ---------------------------------------------------------
    # Select VQC model
    # ---------------------------------------------------------

    selected_model = (
        VQC_MODEL_MAPPING[
            req.model_type
        ]["model"]
    )

    model_name = (
        VQC_MODEL_MAPPING[
            req.model_type
        ]["name"]
    )

    # ---------------------------------------------------------
    # Prediction
    # ---------------------------------------------------------

    try:

        with torch.no_grad():

            probs = (

                selected_model(
                    grid_tensor.to(DEVICE)
                )

                .cpu()
                .numpy()
                .flatten()
            )

    except Exception as e:

        logger.exception(
            "[VQC-Map] Prediction failed"
        )

        raise HTTPException(

            status_code=500,

            detail=(
                "VQC prediction failed: "
                f"{str(e)}"
            )
        )

    # ---------------------------------------------------------
    # Validate prediction length
    # ---------------------------------------------------------

    if len(probs) != len(grid_df):

        raise HTTPException(

            status_code=500,

            detail=(
                f"Number of predictions "
                f"({len(probs)}) does not match "
                f"grid rows ({len(grid_df)})"
            )
        )

    # ---------------------------------------------------------
    # Render map
    # ---------------------------------------------------------

    try:

        png_bytes = (
            render_probability_map_png(

                grid_df,

                probs,

                known_df,

                f"VQC ({model_name})",

                req.target_name
            )
        )

    except Exception as e:

        logger.exception(
            "[VQC-Map] Map rendering failed"
        )

        raise HTTPException(

            status_code=500,

            detail=(
                "Probability map rendering failed: "
                f"{str(e)}"
            )
        )

    # ---------------------------------------------------------
    # Convert PNG bytes -> Base64
    # ---------------------------------------------------------

    image_base64 = base64.b64encode(
        png_bytes
    ).decode("utf-8")

    # ---------------------------------------------------------
    # Return JSON response
    # ---------------------------------------------------------

    return JSONResponse(
        content={
            "target_name": req.target_name,
            "model": f"VQC ({model_name})",
            "image": image_base64
        }
    )

# =============================================================================
# ============================================================================
# 9. VQC AMPLITUDE PROBABILITY MAP
# ============================================================================
# =============================================================================

@app.post(
    "/vqc-amplitude/probability-map"
)
def vqc_amplitude_probability_map(
    req: ProbabilityMapRequest
):

    logger.info(
        f"[VQC-Amplitude-Map] "
        f"target={req.target_name}"
    )

    if vqc_model_amplitude is None:

        raise HTTPException(

            status_code=503,

            detail=(
                "Amplitude VQC weights not found: "
                f"{VQC_AMPLITUDE_WEIGHTS_PATH}"
            )
        )

    try:

        grid_df, known_df = (
            _load_grid_and_known()
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    missing = [

        f

        for f in VQC_AMP_VALID_FEATURES

        if f not in grid_df.columns
    ]

    if missing:

        raise HTTPException(

            status_code=400,

            detail=(
                "Grid CSV is missing columns: "
                f"{missing}"
            )
        )

    grid_X = (
        grid_df[
            VQC_AMP_VALID_FEATURES
        ].values
    )

    grid_X_pad = np.pad(

        grid_X,

        (
            (0, 0),
            (0, 20)
        )
    )

    grid_X_norm = normalize(
        grid_X_pad,
        norm="l2"
    )

    grid_tensor = torch.tensor(
        grid_X_norm,
        dtype=torch.float32
    )

    with torch.no_grad():

        probs = (

            vqc_model_amplitude(
                grid_tensor.to(DEVICE)
            )

            .cpu()
            .numpy()
            .flatten()
        )

    png_bytes = (
        render_probability_map_png(

            grid_df,

            probs,

            known_df,

            "VQC (Amplitude Encoding)",

            req.target_name
        )
    )

    # -----------------------------------------
    # Convert PNG bytes -> Base64
    # -----------------------------------------

    image_base64 = base64.b64encode(
        png_bytes
    ).decode("utf-8")

    # -----------------------------------------
    # Return JSON response
    # -----------------------------------------

    return JSONResponse(
        content={
            "target_name": req.target_name,
            "model": "VQC (Amplitude Encoding)",
            "image": image_base64
        }
    )


# =============================================================================
# ============================================================================
# 10. RANDOM FOREST
# ============================================================================
# =============================================================================
#
# IMPORTANT:
# Change RF_MODEL_PATH below if your RF model has a different filename.
#
# Example:
# Local artifact path: API_files/random_forest_model.pkl
#
# =============================================================================

RF_MODEL_PATH = (
    QAPI_FILES_DIR / "random_forest_model.pkl"
)


# RF feature list.
# If your RF model was trained with a different feature order,
# replace this list with the exact training feature order.

RF_VALID_FEATURES = QSVC_VALID_FEATURES


rf_model = None


if RF_MODEL_PATH.exists():

    try:

        rf_model = joblib.load(
            RF_MODEL_PATH
        )

        logger.info(
            f"[RF] model loaded: "
            f"{RF_MODEL_PATH}"
        )

    except Exception as e:

        logger.warning(
            f"[RF] model loading failed: "
            f"{e}"
        )

else:

    logger.warning(
        f"[RF] Model not found at: "
        f"{RF_MODEL_PATH}"
    )


# =============================================================================
# RF PROBABILITY MAP
# =============================================================================

import base64
from fastapi import HTTPException
from fastapi.responses import JSONResponse


@app.post(
    "/rf/probability-map"
)
def rf_probability_map(
    req: ProbabilityMapRequest
):

    logger.info(
        f"[RF-Map] "
        f"target={req.target_name}"
    )

    if rf_model is None:

        raise HTTPException(

            status_code=503,

            detail=(
                "Random Forest model is not loaded. "
                f"Expected model at: "
                f"{RF_MODEL_PATH}. "
                "Update RF_MODEL_PATH to your actual "
                "RF .pkl file."
            )
        )

    try:

        grid_df, known_df = (
            _load_grid_and_known()
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    missing = [

        f

        for f in RF_VALID_FEATURES

        if f not in grid_df.columns
    ]

    if missing:

        raise HTTPException(

            status_code=400,

            detail=(
                "Grid CSV is missing columns: "
                f"{missing}"
            )
        )

    grid_X = (
        grid_df[
            RF_VALID_FEATURES
        ]
    )

    try:

        # Random Forest classification
        probs = (
            rf_model
            .predict_proba(
                grid_X
            )[:, 1]
        )

    except Exception as e:

        raise HTTPException(

            status_code=500,

            detail=(
                "RF prediction failed. "
                "This usually means the RF model "
                "was trained with a different "
                f"feature set/order. Error: {e}"
            )
        )

    # Generate probability map PNG
    png_bytes = (
        render_probability_map_png(

            grid_df,

            probs,

            known_df,

            "Random Forest",

            req.target_name
        )
    )

    # -----------------------------------------
    # Convert PNG bytes -> Base64
    # -----------------------------------------

    image_base64 = base64.b64encode(
        png_bytes
    ).decode("utf-8")

    # -----------------------------------------
    # Return JSON response
    # -----------------------------------------

    return JSONResponse(
        content={
            "target_name": req.target_name,
            "model": "Random Forest",
            "image": image_base64
        }
    )


# =============================================================================
# ============================================================================
# 11. RF CLASSIFICATION METRICS
# ============================================================================
# =============================================================================

RF_X_TEST_PATH = (
    QAPI_FILES_DIR / r"X_Test_RF.csv"
)

RF_Y_TEST_PATH = (
    QAPI_FILES_DIR / r"y_Test_RF.csv"
)


@app.post(
    "/rf/classification-metrics"
)
def rf_classification_metrics(
    req: MetricsRequest
):

    if rf_model is None:

        raise HTTPException(

            status_code=503,

            detail=(
                "Random Forest model is not loaded. "
                f"Expected: {RF_MODEL_PATH}"
            )
        )

    if not RF_X_TEST_PATH.exists():

        raise HTTPException(

            status_code=404,

            detail=(
                f"RF X test file not found: "
                f"{RF_X_TEST_PATH}"
            )
        )

    if not RF_Y_TEST_PATH.exists():

        raise HTTPException(

            status_code=404,

            detail=(
                f"RF y test file not found: "
                f"{RF_Y_TEST_PATH}"
            )
        )

    try:

        X_test = pd.read_csv(
            RF_X_TEST_PATH
        )

        y_test = pd.read_csv(
            RF_Y_TEST_PATH
        )

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    valid_requested = [

        f

        for f in req.features

        if f in X_test.columns
    ]

    ignored_features = [

        f

        for f in req.features

        if f not in X_test.columns
    ]

    if len(valid_requested) == 0:

        valid_requested = [
            f
            for f in RF_VALID_FEATURES
            if f in X_test.columns
        ]

    try:

        X_selected = (
            X_test[
                valid_requested
            ]
        )

        probabilities = (
            rf_model
            .predict_proba(
                X_selected
            )[:, 1]
        )

        predictions = (
            probabilities >= 0.5
        ).astype(int)

    except Exception as e:

        raise HTTPException(

            status_code=500,

            detail=(
                f"RF prediction failed: {e}"
            )
        )

    y_true = extract_y_true(
        y_test
    )

    accuracy = accuracy_score(
        y_true,
        predictions
    )

    auc_score = roc_auc_score(
        y_true,
        probabilities
    )

    report = round_report(

        classification_report(
            y_true,
            predictions,
            output_dict=True,
            zero_division=0
        )
    )

    return {

        "model_used":
            "Random Forest",

        "classification_accuracy":
            round(float(accuracy), 4),

        "auc_roc":
            round(float(auc_score), 4),

        "classification_report":
            report,

        "ignored_features":
            ignored_features,
    }


# =============================================================================
# ============================================================================
# 12. ROOT / API STATUS
# ============================================================================
# =============================================================================

@app.get("/")
def root():

    return {

        "message":
            "QMine Quantum + Classical Models API Running",

        "status":
            "online",

        "server":
            "FastAPI",

        "endpoints": {

            "rf": {

                "classification_metrics":
                    "/rf/classification-metrics",

                "probability_map":
                    "/rf/probability-map",

                "model_loaded":
                    rf_model is not None,
            },

            "vqc": {

                "classification_metrics":
                    "/vqc/classification-metrics",

                "probability_map":
                    "/vqc/probability-map",

                "available_models":
                    list(
                        VQC_MODEL_MAPPING.keys()
                    ),

                "required_features":
                    VQC_N_QUBITS,
            },

            "qsvc": {

                "classification_metrics":
                    "/qsvc/classification-metrics",

                "probability_map":
                    "/qsvc/probability-map",

                "required_features":
                    QSVC_N_FEATURES,
            },

            "qknn": {

                "classification_metrics":
                    "/qknn/classification-metrics",

                "probability_map":
                    "/qknn/probability-map",

                "required_features":
                    QKNN_N_FEATURES,
            },

            "vqc_amplitude": {

                "classification_metrics":
                    "/vqc-amplitude/classification-metrics",

                "probability_map":
                    "/vqc-amplitude/probability-map",

                "required_features":
                    VQC_AMP_N_FEATURES,

                "weights_loaded":
                    vqc_model_amplitude
                    is not None,
            },

        },

        "files": {

            "prediction_grid":
                str(GRID_CSV_PATH),

            "known_occurrences":
                str(KNOWN_OCCURRENCE_PATH),

            "qapi_directory":
                str(QAPI_FILES_DIR),
        }
    }


# =============================================================================
# ============================================================================
# HEALTH CHECK
# ============================================================================
# =============================================================================

@app.get(
    "/health"
)
def health():

    return {

        "status":
            "healthy",

        "device":
            str(DEVICE),

        "vqc":
            True,

        "qsvc":
            qsvc_model is not None,

        "qknn":
            qknn_model is not None,

        "vqc_amplitude":
            vqc_model_amplitude is not None,

        "random_forest":
            rf_model is not None,
    }


# =============================================================================
# ============================================================================
# START SERVER
# ============================================================================
# =============================================================================

logger.info(
    "================================================"
)

logger.info(
    "QMine API starting..."
)

logger.info(
    f"Host: 0.0.0.0"
)

logger.info(
    f"Port: 6000"
)

logger.info(
    f"Docs: http://192.168.8.19:6000/docs"
)

logger.info(
    "================================================"
)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")