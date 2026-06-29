# Changelog

## 0.6.2

- Added validation-constrained candidate selection: service-level and interval-sharpness feasibility are enforced before cost-based champion selection.
- Added independently tuned baseline-relative policy-scale service constraints.
- Added deterministic M5 cohort offsets and a zero-overlap confirmation protocol.
- Added regression tests for constrained selection, cost-only legacy selector compatibility, and archive-release safety.
- Added expanded M5 development and series-disjoint confirmation configs.
- Excluded `.bak` rollback files and generated `repo_tree.txt` from source-release archives.
- Recorded M5 confirmation evidence: 16.56% held-out WAPE improvement and 18.17% simulated inventory-cost reduction versus seasonal naive on the configured 240-series confirmation cohort, with all release gates passing.

## 0.6.1

- Exclude coverage, environment, editor, and notebook-checkpoint residue from source releases.
- Reject duplicate, encrypted, symlink, traversal, multi-root, and explicit special-file ZIP members.
- Reject partially populated Chronos future-covariate panels instead of silently replacing them.
- Enforce complete Chronos series-by-horizon prediction key coverage and accept numeric or string quantile column names.
- Validate pandas/Spark feature parity before publication and publish Parquet through a rollback-capable staging directory.
- Make `pyproject.toml` the dependency source of truth; requirements and Makefile installation now delegate to it through the shared constraints file.

## 0.6.0

- Added a single-process forecasting-to-SQL runtime smoke entrypoint to avoid repeated numerical interpreter startup while validating the exact connected pipeline.

Reliability and release-hardening update.

- Added transactional pipeline rollback so a failed rerun restores the last valid owned artifacts.
- Moved rollback snapshots outside the repository and excluded orphan backup paths from source packages.
- Added deterministic temporally balanced panel sampling that retains recent audit periods.
- Added model-isolated inventory simulation with shared opening inventory to bound long-lived intermediate state.
- Removed a redundant validation simulation by reusing policy-tuning evidence at scale 1.0.
- Hardened ZIP verification against CRC failures, absolute paths, traversal paths, symlinks, multiple roots, and checksum-sidecar mismatches.
- Added runtime manifest byte-size validation for processed samples.
- Fixed Docker build context to include `LICENSE` before package installation.
- Raised the setuptools build-backend floor to 77.0.3 for PEP 639 `license` and `license-files` support.
- Added optional stage progress logging through `INVENTORY_AI_PROGRESS=1`.
- Expanded regression and contract tests for archive safety, Docker metadata, rollback recovery, audit sampling, and inventory-equivalence behavior.
