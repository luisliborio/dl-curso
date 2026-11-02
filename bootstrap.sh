#!/usr/bin/env bash
# Recria o ambiente do curso num WSL2 (Ubuntu) recém-instalado. Idempotente: pode rodar de novo.
#
# Uso:
#   ./bootstrap.sh                    # detecta GPU NVIDIA → extra "cuda"; senão "cpu"
#   TORCH_VARIANT=cpu ./bootstrap.sh  # força a variante
set -euo pipefail
cd "$(dirname "$0")"

# 0. Repo no filesystem do Windows (/mnt/c) deixa I/O e git muito lentos no WSL2.
case "$PWD" in
  /mnt/*) echo "AVISO: o repo está em $PWD. Prefira clonar em ~/ (filesystem Linux)." ;;
esac

# 1. Pacotes do sistema (instala só o que falta).
need=()
for p in git curl build-essential; do
  dpkg -s "$p" &>/dev/null || need+=("$p")
done
if ((${#need[@]})); then
  sudo apt-get update && sudo apt-get install -y "${need[@]}"
fi

# 2. uv
if ! command -v uv &>/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# 3. Variante do PyTorch. No WSL2, o nvidia-smi vem do driver do Windows em /usr/lib/wsl/lib.
if [[ -z "${TORCH_VARIANT:-}" ]]; then
  smi="$(command -v nvidia-smi || echo /usr/lib/wsl/lib/nvidia-smi)"
  if [[ -x "$smi" ]] && "$smi" -L &>/dev/null; then TORCH_VARIANT=cuda; else TORCH_VARIANT=cpu; fi
fi
echo ">> Variante do PyTorch: $TORCH_VARIANT"

# 4. Python (versão em .python-version) + dependências travadas no uv.lock.
uv python install
uv sync --extra "$TORCH_VARIANT"

# 5. Lembrete de identidade do git (não altera nada sozinho).
if ! git config --global user.email &>/dev/null; then
  echo "Lembrete: git config --global user.name '...' && git config --global user.email '...'"
fi

# 6. Verificação.
uv run python scripts/check_env.py
uv run pytest -q
echo ">> Ambiente pronto. Sempre sincronize com: uv sync --extra $TORCH_VARIANT"
