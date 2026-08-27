# ══════════════════════════════════════════════════════════════════════════════
#  Quantum Models — FastAPI Metrics API  (VQC + QSVC + QKNN)
#  Same request/response contract for all three, same pattern as your
#  working Quantum-API-on_server.ipynb (VQC) endpoint.
# ══════════════════════════════════════════════════════════════════════════════

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
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

from pathlib import Path
from collections import Counter

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from sklearn.preprocessing import normalize
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
)

import nest_asyncio
import uvicorn

# ══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("quantum-api")

# ══════════════════════════════════════════════════════════════════════════════
# APP
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(title="Quantum Models Metrics API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    body = await request.body()

    logger.info(f"➡️ Incoming request: {request.method} {request.url.path}")
    if body:
        logger.info(f"📦 Payload: {body.decode('utf-8', errors='ignore')}")

    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000

    logger.info(
        f"✅ Completed: {request.method} {request.url.path} "
        f"| Status: {response.status_code} "
        f"| Time: {process_time:.2f}ms"
    )
    return response


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def round_report(report: dict) -> dict:
    """Same rounding behaviour as your existing VQC endpoint."""
    for key in report:
        if isinstance(report[key], dict):
            for metric in report[key]:
                if isinstance(report[key][metric], float):
                    report[key][metric] = round(report[key][metric], 2)
        elif isinstance(report[key], float):
            report[key] = round(report[key], 2)
    return report


class MetricsRequest(BaseModel):
    model_type: Optional[str] = None
    features: List[str]


def extract_y_true(y_df: pd.DataFrame) -> np.ndarray:
    """Robustly pull the label column out of a y_test CSV, regardless of
    whether it was saved with an extra index column or a non-'label' name."""
    if y_df.shape[1] == 1:
        return y_df.iloc[:, 0].values
    for col in ["label", "Label", "y", "target"]:
        if col in y_df.columns:
            return y_df[col].values
    # Fallback: last column is almost always the label (first is usually
    # a stray index column from a CSV saved with index=True).
    logger.warning(
        f"y_test CSV has {y_df.shape[1]} columns {list(y_df.columns)} — "
        f"no recognized label column name found, using the last column."
    )
    return y_df.iloc[:, -1].values


# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
#  1) VQC  — unchanged from Quantum-API-on_server.ipynb
# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════

VQC_N_QUBITS = 20
VQC_N_LAYERS = 1
VQC_N_LAYERS_RU = 2
VQC_THRESHOLD = 0.5

VQC_ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts_50"

VQC_X_TEST_PATH = str(Path(__file__).resolve().parent / "X_Test_rfe_new.csv")
VQC_Y_TEST_PATH = str(Path(__file__).resolve().parent / "y_Test_rfe_new.csv")

VQC_BACKEND = "lightning.gpu" if torch.cuda.is_available() else "lightning.qubit"
vqc_dev = qml.device(VQC_BACKEND, wires=VQC_N_QUBITS)


def vqc_ansatz(layer_w):
    for i in range(VQC_N_QUBITS):
        qml.RY(layer_w[i, 0], wires=i)
        qml.RZ(layer_w[i, 1], wires=i)
    for i in range(VQC_N_QUBITS - 1):
        qml.CNOT(wires=[i, i + 1])
    qml.CNOT(wires=[VQC_N_QUBITS - 1, 0])


@qml.qnode(vqc_dev, interface="torch", diff_method="adjoint")
def vqc_circuit_rx(weights, x):
    for i in range(VQC_N_QUBITS):
        qml.RX(x[i], wires=i)
    vqc_ansatz(weights[0])
    return qml.expval(qml.PauliZ(0))


@qml.qnode(vqc_dev, interface="torch", diff_method="adjoint")
def vqc_circuit_ry(weights, x):
    for i in range(VQC_N_QUBITS):
        qml.RY(x[i], wires=i)
    vqc_ansatz(weights[0])
    return qml.expval(qml.PauliZ(0))


@qml.qnode(vqc_dev, interface="torch", diff_method="adjoint")
def vqc_circuit_rz(weights, x):
    for i in range(VQC_N_QUBITS):
        qml.RZ(x[i], wires=i)
    vqc_ansatz(weights[0])
    return qml.expval(qml.PauliZ(0))


@qml.qnode(vqc_dev, interface="torch", diff_method="adjoint")
def vqc_circuit_learnable(weights, enc, x):
    for i in range(VQC_N_QUBITS):
        qml.RY(enc[i, 0] * x[i] + enc[i, 1], wires=i)
    vqc_ansatz(weights[0])
    return qml.expval(qml.PauliZ(0))


@qml.qnode(vqc_dev, interface="torch", diff_method="adjoint")
def vqc_circuit_reupload(weights, x):
    for r in range(VQC_N_LAYERS_RU):
        for i in range(VQC_N_QUBITS):
            qml.RY(x[i], wires=i)
        vqc_ansatz(weights[r])
    return qml.expval(qml.PauliZ(0))


class AngleVQC(nn.Module):
    def __init__(self, circuit, n_layers=VQC_N_LAYERS):
        super().__init__()
        self.circuit = circuit
        self.weights = nn.Parameter(
            torch.tensor(
                np.random.uniform(-np.pi, np.pi, (n_layers, VQC_N_QUBITS, 2)),
                dtype=torch.float32,
            )
        )

    def forward(self, X_batch):
        out = torch.stack([self.circuit(self.weights, x) for x in X_batch])
        return (out.float() + 1.0) / 2.0


class LearnableVQC(nn.Module):
    def __init__(self, n_layers=VQC_N_LAYERS):
        super().__init__()
        self.weights = nn.Parameter(
            torch.tensor(
                np.random.uniform(-np.pi, np.pi, (n_layers, VQC_N_QUBITS, 2)),
                dtype=torch.float32,
            )
        )
        enc = torch.zeros(VQC_N_QUBITS, 2, dtype=torch.float32)
        enc[:, 0] = 1.0
        self.enc = nn.Parameter(enc)

    def forward(self, X_batch):
        out = torch.stack(
            [vqc_circuit_learnable(self.weights, self.enc, x) for x in X_batch]
        )
        return (out.float() + 1.0) / 2.0


class ReuploadVQC(nn.Module):
    def __init__(self, n_layers=VQC_N_LAYERS_RU):
        super().__init__()
        self.weights = nn.Parameter(
            torch.tensor(
                np.random.uniform(-np.pi, np.pi, (n_layers, VQC_N_QUBITS, 2)),
                dtype=torch.float32,
            )
        )

    def forward(self, X_batch):
        out = torch.stack([vqc_circuit_reupload(self.weights, x) for x in X_batch])
        return (out.float() + 1.0) / 2.0


def vqc_load_model(model_obj, path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Weights not found: {path}")
    model_obj.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    model_obj.eval()
    return model_obj.to(DEVICE)


vqc_model_rx = vqc_load_model(AngleVQC(vqc_circuit_rx), VQC_ARTIFACT_DIR / "rx_encoding_weights.pt")
vqc_model_ry = vqc_load_model(AngleVQC(vqc_circuit_ry), VQC_ARTIFACT_DIR / "ry_encoding_weights.pt")
vqc_model_rz = vqc_load_model(AngleVQC(vqc_circuit_rz), VQC_ARTIFACT_DIR / "rz_encoding_weights.pt")
vqc_model_learnable = vqc_load_model(LearnableVQC(), VQC_ARTIFACT_DIR / "learnable_encoding_weights.pt")
vqc_model_reupload = vqc_load_model(ReuploadVQC(), VQC_ARTIFACT_DIR / "data_re-uploading_weights.pt")

VQC_MODEL_MAPPING = {
    "rx": {"model": vqc_model_rx, "name": "RX Encoding"},
    "ry": {"model": vqc_model_ry, "name": "RY Encoding"},
    "rz": {"model": vqc_model_rz, "name": "RZ Encoding"},
    "learnable": {"model": vqc_model_learnable, "name": "Learnable Encoding"},
    "data_reuploading": {"model": vqc_model_reupload, "name": "Data Re-uploading"},
}


@app.post("/vqc/classification-metrics")
def vqc_classification_metrics(req: MetricsRequest):
    logger.info(f"[VQC] Model requested: {req.model_type}")

    if req.model_type not in VQC_MODEL_MAPPING:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model_type. Use one of: {list(VQC_MODEL_MAPPING.keys())}",
        )

    try:
        X_test = pd.read_csv(VQC_X_TEST_PATH)
        y_test = pd.read_csv(VQC_Y_TEST_PATH)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    valid_requested = [f for f in req.features if f in X_test.columns]
    ignored_features = [f for f in req.features if f not in X_test.columns]

    if len(valid_requested) != VQC_N_QUBITS:
        raise HTTPException(
            status_code=400,
            detail=f"Need exactly {VQC_N_QUBITS} valid features. Got {len(valid_requested)}.",
        )

    X_selected = X_test[valid_requested]
    X_tensor = torch.tensor(X_selected.values, dtype=torch.float32)

    selected_model = VQC_MODEL_MAPPING[req.model_type]["model"]
    model_name = VQC_MODEL_MAPPING[req.model_type]["name"]

    with torch.no_grad():
        probabilities = selected_model(X_tensor.to(DEVICE)).cpu().numpy()

    predictions = (probabilities >= VQC_THRESHOLD).astype(int)

    y_true = extract_y_true(y_test)
    y_pred = predictions.flatten()
    y_prob = probabilities.flatten()

    if len(y_true) != len(y_pred):
        raise HTTPException(
            status_code=400,
            detail=(
                f"y_test has {len(y_true)} rows but predictions have {len(y_pred)} — "
                f"check that {VQC_Y_TEST_PATH} has one row per test sample and only "
                f"a single label column."
            ),
        )

    accuracy = accuracy_score(y_true, y_pred)
    auc_score = roc_auc_score(y_true, y_prob)
    report = round_report(
        classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    )

    logger.info(f"[VQC] model={model_name} acc={accuracy:.4f} auc={auc_score:.4f}")

    return {
        "model_used": model_name,
        "classification_accuracy": round(float(accuracy), 4),
        "auc_roc": round(float(auc_score), 4),
        "classification_report": report,
        "ignored_features": ignored_features,
    }


# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
#  2) QSVC — amplitude-encoded precomputed-kernel SVM (6 qubits, 44 features)
# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════

QSVC_N_QUBITS = 6
QSVC_AMP_DIM = 2 ** QSVC_N_QUBITS  # 64

# Flat artifact folder on the server — everything (VQC weights aside) lives here
QAPI_FILES_DIR = str(Path(__file__).resolve().parent)

QSVC_N_FEATURES = 44   # UPDATED: retrained QSVC now uses the same feature set as QKNN

QSVC_X_TEST_PATH = f"{QAPI_FILES_DIR}/X_Test_QSVC_38.csv"
QSVC_Y_TEST_PATH = f"{QAPI_FILES_DIR}/y_Test_QSVC_38.csv"

# Needed to rebuild the precomputed kernel against the exact training set the
# model was fit on. Export this once from the training notebook (see the
# export snippet) and place it on the server.
QSVC_XTRAIN_NORM_PATH = f"{QAPI_FILES_DIR}/X_train_norm_38.npy"
QSVC_MODEL_PATH = f"{QAPI_FILES_DIR}/qsvc_38.pkl"

# UPDATED real feature names/order (from the retrained QSVC export) — now the
# same 44 features/order as QKNN. Requests must supply exactly these 44 names.
QSVC_VALID_FEATURES = [
    "fault_dist", "fold_dist", "PC1", "PC2", "PC3", "elevation1", "observed_1",
    "theoreti_1", "bouguer__1", "BG", "BG_Residual", "BG_1VD", "BG_THD", "BG_TDR",
    "observed_m", "igrf_nt", "magnetic_a", "MAG_TF_raw", "MAG_TF", "MAG_Residual",
    "MAG_1VZ", "MAG_AS", "Amphibolite–Metabasic rock",
    "Brecciated ferruginous quartzite", "Calc-gneiss",
    "Carbonaceous phyllite with schist", "Conglomerate", "Dolomitic marble",
    "Fine sand, silt and clay", "Gneiss", "Granite", "Granite gneiss",
    "Graphite schist", "Meta-dolerite", "Other", "Pegmatite", "Phyllite",
    "Quartz vein", "Quartzite", "Quartzite with Phyllite", "Schist",
    "Calc-silicate", "Carbonate rock", "Migmatite",
]


def qsvc_normalize_amplitude(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x)
    return x / norm if norm > 0 else x


def qsvc_amplitude_map(x: np.ndarray) -> QuantumCircuit:
    qc = QuantumCircuit(QSVC_N_QUBITS)
    padded = np.zeros(QSVC_AMP_DIM)
    padded[: len(x)] = x[:QSVC_N_FEATURES]
    qc.initialize(padded, range(QSVC_N_QUBITS))
    return qc


def qsvc_compute_kernel(X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
    """
    Amplitude-encoding fidelity kernel, computed in closed form instead of via
    per-pair quantum circuit simulation.

    For amplitude encoding, the two prepared states are |psi1> = sum(x1_i |i>)
    and |psi2> = sum(x2_i |i>) (real unit vectors, zero-padded). Fidelity is
    exactly |<psi1|psi2>|^2 = (x1 . x2)^2 for real amplitudes — this is not an
    approximation, it's the same quantity qiskit's Statevector.inner() computes,
    just via matrix multiplication instead of simulating a circuit per pair.
    This turns an O(n1 * n2) loop of circuit simulations into one BLAS matmul.
    """
    return (X1 @ X2.T) ** 2


qsvc_model = joblib.load(QSVC_MODEL_PATH)
qsvc_X_train_norm = np.load(QSVC_XTRAIN_NORM_PATH)


@app.post("/qsvc/classification-metrics")
def qsvc_classification_metrics(req: MetricsRequest):
    logger.info("[QSVC] request received")

    try:
        X_test = pd.read_csv(QSVC_X_TEST_PATH)
        y_test = pd.read_csv(QSVC_Y_TEST_PATH)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    valid_requested = [f for f in req.features if f in X_test.columns]
    ignored_features = [f for f in req.features if f not in X_test.columns]

    if len(valid_requested) != QSVC_N_FEATURES:
        raise HTTPException(
            status_code=400,
            detail=f"Need exactly {QSVC_N_FEATURES} valid features. Got {len(valid_requested)}.",
        )

    X_selected = X_test[valid_requested].values
    X_selected_norm = np.array([qsvc_normalize_amplitude(x) for x in X_selected])

    K_test = qsvc_compute_kernel(X_selected_norm, qsvc_X_train_norm)

    y_pred = qsvc_model.predict(K_test)
    y_score = qsvc_model.decision_function(K_test)   # kernel='precomputed' has no predict_proba

    y_true = extract_y_true(y_test)

    if len(y_true) != len(y_pred):
        raise HTTPException(
            status_code=400,
            detail=(
                f"y_test has {len(y_true)} rows but predictions have {len(y_pred)} — "
                f"check that {QSVC_Y_TEST_PATH} has one row per test sample and only "
                f"a single label column."
            ),
        )

    accuracy = accuracy_score(y_true, y_pred)
    auc_score = roc_auc_score(y_true, y_score)
    report = round_report(
        classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    )

    logger.info(f"[QSVC] acc={accuracy:.4f} auc={auc_score:.4f}")

    return {
        "model_used": "QSVC (Amplitude-Encoded Kernel)",
        "classification_accuracy": round(float(accuracy), 4),
        "auc_roc": round(float(auc_score), 4),
        "classification_report": report,
        "ignored_features": ignored_features,
    }


# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
#  3) QKNN — quantum fidelity K-Nearest-Neighbours (6 qubits, 44 features)
# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════

QKNN_N_FEATURES = 44
QKNN_N_QUBITS = 6
QKNN_AMP_DIM = 2 ** QKNN_N_QUBITS  # 64

# CONFIRMED: QKNN uses its own dataset/split (mastergrid_clean_15k.csv), NOT
# the same file as QSVC (which uses a different, downsampled dataset).
QKNN_X_TEST_PATH = f"{QAPI_FILES_DIR}/X_Test_QKNN.csv"
QKNN_Y_TEST_PATH = f"{QAPI_FILES_DIR}/y_Test_QKNN.csv"

QKNN_MODEL_PATH = f"{QAPI_FILES_DIR}/qknn_model.pkl"   # CONFIRMED, already saved in training nb

# CONFIRMED real feature names/order (from balanced_data.drop(columns=["x","y","label"])
# on mastergrid_clean_15k.csv). Requests must supply exactly these 44 names.
QKNN_VALID_FEATURES = [
    "fault_dist", "fold_dist", "PC1", "PC2", "PC3", "elevation1", "observed_1",
    "theoreti_1", "bouguer__1", "BG", "BG_Residual", "BG_1VD", "BG_THD", "BG_TDR",
    "observed_m", "igrf_nt", "magnetic_a", "MAG_TF_raw", "MAG_TF", "MAG_Residual",
    "MAG_1VZ", "MAG_AS", "Amphibolite–Metabasic rock",
    "Brecciated ferruginous quartzite", "Calc-gneiss",
    "Carbonaceous phyllite with schist", "Conglomerate", "Dolomitic marble",
    "Fine sand, silt and clay", "Gneiss", "Granite", "Granite gneiss",
    "Graphite schist", "Meta-dolerite", "Other", "Pegmatite", "Phyllite",
    "Quartz vein", "Quartzite", "Quartzite with Phyllite", "Schist",
    "Calc-silicate", "Carbonate rock", "Migmatite",
]

qknn_dev = qml.device("lightning.qubit", wires=QKNN_N_QUBITS)


@qml.qnode(qknn_dev)
def qknn_fidelity_circuit(x1, x2):
    qml.StatePrep(x1, wires=range(QKNN_N_QUBITS))
    qml.adjoint(qml.StatePrep)(x2, wires=range(QKNN_N_QUBITS))
    return qml.probs(wires=range(QKNN_N_QUBITS))


def qknn_quantum_distance(x1, x2) -> float:
    # Kept for reference / single-pair debugging — no longer used in the hot
    # path (see _compute_distances below), which vectorizes this instead.
    fidelity = float(qknn_fidelity_circuit(x1, x2)[0])
    return 1.0 - fidelity


class QuantumKNN:
    def __init__(self, k: int = 5):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def _compute_distances(self, x):
        """
        Vectorized replacement for the per-pair PennyLane fidelity circuit.

        qknn_fidelity_circuit prepares |x1>, applies StatePrep(x2)^-1, and
        reads probs()[0] — i.e. the probability of measuring all-zeros, which
        for two real, normalized amplitude-encoded states is exactly
        |<x2|x1>|^2 = (x1 . x2)^2. So distance = 1 - (x1 . x2)^2, computable
        for ALL training rows at once via one matrix-vector product instead
        of one circuit simulation per training row.
        """
        fidelities = (self.X_train @ x) ** 2
        return 1.0 - fidelities

    def predict_single(self, x):
        distances = self._compute_distances(x)
        knn_idx = np.argsort(distances)[: self.k]
        knn_labels = self.y_train[knn_idx]
        return Counter(knn_labels).most_common(1)[0][0]

    def predict(self, X):
        return np.array([self.predict_single(x) for x in X])

    def predict_proba(self, X):
        probas = []
        for x in X:
            distances = self._compute_distances(x)
            knn_idx = np.argsort(distances)[: self.k]
            knn_dists = distances[knn_idx] + 1e-10
            weights = 1.0 / knn_dists
            weights /= weights.sum()
            prob1 = sum(w for w, lbl in zip(weights, self.y_train[knn_idx]) if lbl == 1)
            probas.append([1 - prob1, prob1])
        return np.array(probas)


def qknn_load_model(path: str) -> QuantumKNN:
    with open(path, "rb") as f:
        state = pickle.load(f)
    model = QuantumKNN(k=state["k"])
    model.X_train = state["X_train"]
    model.y_train = state["y_train"]
    return model


qknn_model = qknn_load_model(QKNN_MODEL_PATH)


@app.post("/qknn/classification-metrics")
def qknn_classification_metrics(req: MetricsRequest):
    logger.info("[QKNN] request received")

    try:
        X_test = pd.read_csv(QKNN_X_TEST_PATH)
        y_test = pd.read_csv(QKNN_Y_TEST_PATH)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    valid_requested = [f for f in req.features if f in X_test.columns]
    ignored_features = [f for f in req.features if f not in X_test.columns]

    if len(valid_requested) != QKNN_N_FEATURES:
        raise HTTPException(
            status_code=400,
            detail=f"Need exactly {QKNN_N_FEATURES} valid features. Got {len(valid_requested)}.",
        )

    X_selected = X_test[valid_requested].values
    X_pad = np.pad(X_selected, ((0, 0), (0, QKNN_AMP_DIM - QKNN_N_FEATURES)))
    X_norm = normalize(X_pad, norm="l2")

    y_pred = qknn_model.predict(X_norm)
    y_score = qknn_model.predict_proba(X_norm)[:, 1]

    y_true = extract_y_true(y_test)

    if len(y_true) != len(y_pred):
        raise HTTPException(
            status_code=400,
            detail=(
                f"y_test has {len(y_true)} rows but predictions have {len(y_pred)} — "
                f"check that {QKNN_Y_TEST_PATH} has one row per test sample and only "
                f"a single label column."
            ),
        )

    accuracy = accuracy_score(y_true, y_pred)
    auc_score = roc_auc_score(y_true, y_score)
    report = round_report(
        classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    )

    logger.info(f"[QKNN] acc={accuracy:.4f} auc={auc_score:.4f}")

    return {
        "model_used": f"QKNN (k={qknn_model.k}, Fidelity Distance)",
        "classification_accuracy": round(float(accuracy), 4),
        "auc_roc": round(float(auc_score), 4),
        "classification_report": report,
        "ignored_features": ignored_features,
    }


# ══════════════════════════════════════════════════════════════════════════════
# ROOT
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {
        "message": "Quantum Models Metrics API Running",
        "endpoints": {
            "vqc": {
                "path": "/vqc/classification-metrics",
                "available_models": list(VQC_MODEL_MAPPING.keys()),
                "required_features": VQC_N_QUBITS,
            },
            "qsvc": {
                "path": "/qsvc/classification-metrics",
                "required_features": QSVC_N_FEATURES,
            },
            "qknn": {
                "path": "/qknn/classification-metrics",
                "required_features": QKNN_N_FEATURES,
            },
        },
    }

# ══════════════════════════════════════════════════════════════════════════════
#  RUN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    nest_asyncio.apply()
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8080")),
        loop="asyncio",
        log_level="info",
        access_log=True,
    )
    server = uvicorn.Server(config)
    import asyncio
    asyncio.run(server.serve())
