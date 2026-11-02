# Módulos

Um diretório por módulo do plano (`m1-transformer/`, `m2-treino-eficiente/`, ...).
Código reaproveitável entre módulos sobe para `src/dlcurso/`.

Convenção dentro de cada módulo:
- `README.md` — objetivo, como rodar, resultados (tabelas com média ± desvio entre seeds)
- `*.py` — código executável; notebooks só para exploração
- Configs com perfil `small` (roda em CPU no notebook) e `full` (GPU)
