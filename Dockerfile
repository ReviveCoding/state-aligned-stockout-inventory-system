FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    PYTHONHASHSEED=0

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN python -m pip install --upgrade pip && python -m pip install .
COPY configs ./configs
COPY scripts ./scripts
COPY sql ./sql
COPY tests ./tests
COPY *.md Makefile ./
RUN python -m compileall -q src scripts
CMD ["python", "scripts/run_pipeline.py", "--config", "configs/smoke.yaml"]
