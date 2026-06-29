from __future__ import annotations

import json
from pathlib import Path

from inventory_ai.utils.io import atomic_write_text


def write_release_report(path: str | Path, release: dict, metrics: dict) -> None:
    selected = metrics.get("selected_model")
    test_metrics = metrics.get("test_forecast_metrics", {})
    candidate = test_metrics.get("production_candidate", {})
    inventory = metrics.get("inventory_summary", {}).get("production_candidate", {})
    recovery = metrics.get("recovery_diagnostics", {})
    lifecycle = metrics.get("lifecycle_metrics", {})
    failed = [name for name, passed in release["checks"].items() if not passed]
    text = f"""# Release Qualification Report

## Decision

**{release['gate_status']}**

Selected source model: **{selected}**

Failed or review checks: {', '.join(failed) if failed else 'None'}

## Forecast evidence

- Candidate WAPE: {candidate.get('wape')}
- Candidate signed bias: {candidate.get('signed_bias')}
- Candidate 80% coverage: {candidate.get('coverage_80')}
- Candidate mean interval width: {candidate.get('mean_interval_width')}

## Censoring recovery

- Controlled rows: {recovery.get('n_controlled')}
- Raw MAE: {recovery.get('raw_mae')}
- Recovered MAE: {recovery.get('recovered_mae')}
- Raw bias: {recovery.get('raw_bias')}
- Recovered bias: {recovery.get('recovered_bias')}

## Inventory decision evidence

- Candidate total cost: {inventory.get('total_cost')}
- Candidate fill rate: {inventory.get('fill_rate')}
- Candidate stockout rate: {inventory.get('stockout_rate')}
- Cost win rate versus seasonal naive: {metrics.get('cost_win_rate')}

## Lifecycle benchmark

- State macro-F1: {lifecycle.get('state_macro_f1')}
- LADT MAE: {lifecycle.get('ladt_mae')}
- LADT Spearman correlation: {lifecycle.get('ladt_spearman')}

## Reliability checks

```json
{json.dumps(release['checks'], indent=2)}
```

## Claim boundaries

- M5 is used for hierarchical retail forecasting mechanics, price/calendar covariates, and controlled censoring experiments. It does not provide true inventory or lifecycle labels.
- Controlled-stockout recovery accuracy is measured only where latent demand is preserved by construction.
- LADT recovery is evaluated only on the controlled variable-duration lifecycle benchmark.
- Inventory savings are offline simulation results under the documented cost and lead-time assumptions.
"""
    atomic_write_text(path, text)
