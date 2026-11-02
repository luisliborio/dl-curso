# dl-curso

Código do curso contínuo de Deep Learning. O ambiente é **recriado, não copiado**:
qualquer WSL2 chega ao mesmo estado com `git clone` + `./bootstrap.sh`.

## Máquina nova (WSL2)

```bash
cd ~                                  # filesystem Linux, nunca /mnt/c
git clone git@github.com:<usuario>/dl-curso.git
cd dl-curso
./bootstrap.sh                        # detecta GPU → cpu ou cuda; roda check e testes
```

## Dia a dia

```bash
uv sync --extra cpu        # ou --extra cuda. Atenção: `uv sync` SEM extra remove o torch
uv run python scripts/check_env.py
uv run pytest -q
uv add <pacote>            # adiciona dependência e atualiza o uv.lock (commitar os dois)
```

No VS Code: selecionar o interpretador `.venv/bin/python`.

## Colab (GPU de reserva)

```python
!git clone https://github.com/<usuario>/dl-curso.git   # repo privado: usar token via Colab Secrets
%cd dl-curso
!pip install -q -e .        # usa o torch que já vem no Colab
from dlcurso.env import env_report; env_report()
```

## Estrutura

```
src/dlcurso/   código compartilhado (env: device, seeds, caminhos, commit hash)
modulos/       um diretório por módulo do plano
scripts/       utilitários (check_env.py)
tests/         testes de fumaça
data/          datasets (fora do Git; DLCURSO_DATA para mudar o local)
artifacts/     checkpoints e runs (fora do Git; DLCURSO_ARTIFACTS)
```

## Regras de portabilidade

- Tudo que define o ambiente está versionado: `pyproject.toml`, `uv.lock`, `.python-version`.
- Dados são baixados por script; nunca commitados.
- Código nunca assume `cuda`: usar `dlcurso.env.get_device()`.
- Trocar a versão de CUDA = mudar uma URL no `pyproject.toml` (índice `pytorch-cuda`) e rodar `uv lock`.
- Docker/Dev Container: planejado para o M3 (multi-GPU/cloud).
