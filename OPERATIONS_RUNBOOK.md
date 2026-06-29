# Operations Runbook

## Supported Python

Python 3.11, 3.12, and 3.13 are covered by the GitHub Actions matrix.

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .[dev] -c constraints/base.txt
```

Set deterministic thread limits in constrained environments:

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONHASHSEED=0
```

## Progress logging and failed-run recovery

```bash
INVENTORY_AI_PROGRESS=1 python scripts/run_pipeline.py --config configs/smoke.yaml
```

Progress messages are written to stderr. Before a run, owned outputs are copied to a temporary location outside the repository. If an exception occurs, partial owned files are removed and the previous valid files are restored. Externally terminated processes cannot leave packageable backup directories inside the repository.

## Connected runtime smoke

```bash
python scripts/run_runtime_smoke.py --config configs/smoke.yaml
```

This executes forecasting and SQL marts in one fresh process and returns a concise gate/row-count summary. The archive runtime verifier uses the same entrypoint.

## Synthetic verification

```bash
make verify
```

This runs, as independent commands:

1. full pytest suite
2. synthetic pipeline
3. SQL marts
4. manifest, release-gate, wheel, isolated import, and CLI verification

## M5 verification

Place the archive at the path configured in `configs/m5_smoke.yaml`, currently `/mnt/data/m5-forecasting-accuracy.zip`.

```bash
make verify-m5
```

`source: m5` raises a hard error for missing, corrupt, or schema-invalid archives. `source: auto` may fall back only when the archive is absent and `allow_fallback: true`.

## Output ownership

Each run owns separate paths:

```text
reports/<run_name>/
artifacts/<run_name>/
data/processed/<run_name>/
```

The forecasting stage rewrites only its declared artifacts. The SQL stage adds its own marts and provenance. Unrelated files are not added to the run manifest.

## Failure and fallback behavior

| Failure | Behavior |
|---|---|
| Chronos optional dependency unavailable | core seasonal/TSB/GBM/router pipeline remains runnable; optional command returns install guidance |
| PySpark unavailable | core pandas path remains runnable; Spark command returns install guidance |
| advanced model fails validation | seasonal naive remains an eligible fallback |
| candidate lowers cost but materially harms fill rate | candidate fails the decision gate |
| candidate improves average WAPE but harms a slice | candidate fails or is replaced by validation-only routing |
| SQL is run before the pipeline | explicit missing-input error |
| M5 ZIP is corrupt | hard error; no silent synthetic substitution |
| critical data or forecast contract fails | pipeline stops or release returns FAIL |
| noncritical empirical gate fails | release returns ITERATE |

## Reproducibility controls

- deterministic seeds and row-keyed randomization
- causal temporal features
- disjoint calibration, validation, and test origins
- stable per-series posterior/scenario sampling
- atomic JSON/CSV writes
- idempotent run-specific output ownership
- config/environment/run manifests
- SHA-256 and byte-size validation
- wheel single-source version and isolated import
- source archive file-set manifest

## Packaging and fresh-extraction verification

```bash
make package
make verify-package-static
make verify-package-tests
make verify-package-runtime
```

- `package` creates the ZIP, checksum sidecar, and validates the packaged file set and hashes.
- `verify-package-static` rechecks archive integrity without running models.
- `verify-package-tests` extracts to a fresh directory and runs the test suite.
- `verify-package-runtime` uses a separate fresh extraction for pipeline, SQL, release-gate, and run-manifest checks.

The verification stages are separate to avoid duplicated numerical model fitting in a single long-lived process.

## Optional capabilities

```bash
make capabilities
```

Chronos:

```bash
python -m pip install -e .[chronos]
make chronos-zero-shot
make chronos-lora
```

Spark:

```bash
python -m pip install -e .[spark]
make spark-features
```
