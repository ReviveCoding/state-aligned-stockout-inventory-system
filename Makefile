.PHONY: install run smoke spark-features chronos-zero-shot chronos-lora m5-smoke auto-local test test-m5 sql sql-m5 verify verify-m5 package verify-package-static verify-package-tests verify-package-runtime build clean capabilities

install:
	python -m pip install -r requirements.txt

run:
	$(MAKE) smoke
	$(MAKE) sql
	$(MAKE) test

smoke:
	python scripts/run_pipeline.py --config configs/smoke.yaml

m5-smoke:
	python scripts/run_pipeline.py --config configs/m5_smoke.yaml

auto-local:
	python scripts/run_pipeline.py --config configs/auto_local.yaml

test:
	python -m pytest -q

test-m5:
	RUN_M5_TESTS=1 python -m pytest -q tests/integration/test_m5_optional.py

sql:
	python scripts/run_sql_marts.py --config configs/smoke.yaml

sql-m5:
	python scripts/run_sql_marts.py --config configs/m5_smoke.yaml

capabilities:
	python scripts/check_optional_capabilities.py

spark-features:
	python scripts/run_spark_features.py --config configs/smoke.yaml

chronos-zero-shot:
	python scripts/run_chronos.py --config configs/m5_smoke.yaml --mode zero-shot --device-map cuda

chronos-lora:
	python scripts/run_chronos.py --config configs/m5_smoke.yaml --mode lora --device-map cuda

verify:
	python -m pytest -q
	python scripts/run_pipeline.py --config configs/smoke.yaml
	python scripts/run_sql_marts.py --config configs/smoke.yaml
	python scripts/verify_repository.py --config configs/smoke.yaml --skip-pipeline --skip-tests

verify-m5:
	RUN_M5_TESTS=1 python -m pytest -q tests/integration/test_m5_optional.py
	python scripts/run_pipeline.py --config configs/m5_smoke.yaml
	python scripts/run_sql_marts.py --config configs/m5_smoke.yaml
	python scripts/verify_repository.py --config configs/m5_smoke.yaml --skip-pipeline --skip-tests --skip-build

build:
	python -m build

package:
	mkdir -p dist
	python scripts/package_release.py --output dist/state_aligned_stockout_inventory_system.zip

verify-package-static:
	python scripts/verify_archive.py --archive dist/state_aligned_stockout_inventory_system.zip --mode static

verify-package-tests:
	python scripts/verify_archive.py --archive dist/state_aligned_stockout_inventory_system.zip --mode tests

verify-package-runtime:
	python scripts/verify_archive.py --archive dist/state_aligned_stockout_inventory_system.zip --mode runtime

clean:
	python -c "from pathlib import Path; import shutil; [shutil.rmtree(p, ignore_errors=True) for p in Path('.').rglob('__pycache__')]; [p.unlink() for p in Path('.').rglob('*.pyc') if p.exists()]"
	rm -rf .pytest_cache build dist *.egg-info src/*.egg-info
	rm -f .coverage .coverage.*
