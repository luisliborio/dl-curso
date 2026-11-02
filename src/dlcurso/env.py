"""Ambiente agnóstico de máquina: device, seeds, caminhos e relatório do ambiente.

O mesmo código roda no notebook (CPU), na workstation (CUDA) e no Colab.
Overrides por variável de ambiente:
    DLCURSO_DEVICE     força o device ("cpu", "cuda", "cuda:1", ...)
    DLCURSO_DATA       diretório de datasets   (padrão: <repo>/data)
    DLCURSO_ARTIFACTS  checkpoints, runs, logs (padrão: <repo>/artifacts)
"""

from __future__ import annotations

import os
import platform
import random
import subprocess
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]


def _dir_from_env(var: str, default: Path) -> Path:
    path = Path(os.environ.get(var, default)).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def data_dir() -> Path:
    return _dir_from_env("DLCURSO_DATA", REPO_ROOT / "data")


def artifacts_dir() -> Path:
    return _dir_from_env("DLCURSO_ARTIFACTS", REPO_ROOT / "artifacts")


def get_device() -> torch.device:
    """cuda > mps > cpu, salvo override por DLCURSO_DEVICE."""
    if forced := os.environ.get("DLCURSO_DEVICE"):
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Fixa seeds de random/numpy/torch. `deterministic=True` também força kernels determinísticos
    (mais lento; use para depurar ou em ablações onde a variância de kernel importa)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # cobre todas as GPUs
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def git_commit() -> str | None:
    """Hash do commit atual (+ marca '-dirty' se houver mudanças). Logue isso em todo experimento."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = subprocess.run(
            ["git", "diff", "--quiet", "HEAD"], check=False, cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
        ).returncode != 0
        return sha + ("-dirty" if dirty else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def env_report() -> dict:
    """Resumo do ambiente — para imprimir no início de cada run e salvar junto dos resultados."""
    device = get_device()
    report = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "device": str(device),
        "cpu_threads": torch.get_num_threads(),
        "git_commit": git_commit(),
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        report |= {
            "gpu": props.name,
            "gpu_mem_gb": round(props.total_memory / 1024**3, 1),
            "compute_capability": f"{props.major}.{props.minor}",
            "bf16": torch.cuda.is_bf16_supported(),
        }
    return report
