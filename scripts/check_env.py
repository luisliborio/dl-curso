"""Verificação rápida do ambiente: imprime o relatório e mede um matmul no device padrão."""

import time

import torch

from dlcurso.env import env_report, get_device


def main() -> None:
    for k, v in env_report().items():
        print(f"{k:>20}: {v}")

    device = get_device()
    n = 2048
    a = torch.randn(n, n, device=device)
    b = torch.randn(n, n, device=device)
    for _ in range(3):  # aquecimento
        a @ b
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    reps = 10
    for _ in range(reps):
        a @ b
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / reps
    print(f"{'matmul fp32 2048²':>20}: {2 * n**3 / dt / 1e12:.2f} TFLOP/s")


if __name__ == "__main__":
    main()
