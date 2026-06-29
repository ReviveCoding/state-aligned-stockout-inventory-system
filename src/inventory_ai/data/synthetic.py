from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SyntheticConfig:
    n_series: int = 24
    history_days: int = 140
    seed: int = 42


def make_synthetic_retail(cfg: SyntheticConfig) -> pd.DataFrame:
    """Generate heterogeneous retail series with known latent demand.

    The generator deliberately includes smooth trends, weekly seasonality,
    promotions, structural breaks, intermittent series, and natural stockouts.
    """
    rng = np.random.default_rng(cfg.seed)
    rows: list[dict] = []
    for sid in range(cfg.n_series):
        base = rng.uniform(5, 32)
        weekly_amp = rng.uniform(0.05, 0.35)
        price0 = rng.uniform(2.5, 14.0)
        elasticity = rng.uniform(0.3, 1.2)
        trend = rng.normal(0.015, 0.035)
        intermittent = sid % 7 == 0
        break_day = int(rng.integers(cfg.history_days // 3, 2 * cfg.history_days // 3))
        break_mult = rng.uniform(0.7, 1.35)
        category = f"cat_{sid % 4}"
        store = f"store_{sid % 5}"
        promo_period = int(rng.integers(17, 29))
        for t in range(cfg.history_days):
            dow = t % 7
            promo = int((t + sid) % promo_period == 0)
            lifecycle = 1.0 + 0.28 * np.tanh((t - 24) / 18) - 0.16 * np.tanh((t - 105) / 16)
            structural = 1.0 if t < break_day else break_mult
            seasonal = 1.0 + weekly_amp * np.sin(2 * np.pi * dow / 7)
            price = price0 * (0.90 if promo else 1.0) * (1 + rng.normal(0, 0.012))
            price_effect = (price0 / max(price, 0.1)) ** elasticity
            latent_mean = max(0.05, (base + trend * t) * lifecycle * structural * seasonal * price_effect)
            latent_mean += promo * base * rng.uniform(0.25, 0.55)
            if intermittent and rng.random() < 0.48:
                latent = 0.0
            else:
                shape = max(1.0, latent_mean / 2.0)
                latent = float(rng.gamma(shape=shape, scale=max(0.1, latent_mean / shape)))
            stockout_prob = min(0.55, 0.03 + 0.12 * promo + 0.15 * (latent > 1.4 * base))
            stockout = int(rng.random() < stockout_prob and latent > 0)
            available = latent * rng.uniform(0.35, 0.85) if stockout else latent + rng.uniform(1, 6)
            sales = min(latent, available)
            rows.append(
                {
                    "series_id": f"item_{sid:03d}_{store}",
                    "date_idx": t,
                    "sales": float(max(0.0, sales)),
                    "demand": float(max(0.0, latent)),
                    "stockout": stockout,
                    "price": float(max(0.01, price)),
                    "promo": promo,
                    "category": category,
                    "store": store,
                    "dow": dow,
                }
            )
    return pd.DataFrame(rows).sort_values(["series_id", "date_idx"]).reset_index(drop=True)


def make_lifecycle_benchmark(cfg: SyntheticConfig) -> pd.DataFrame:
    """Generate variable-duration lifecycle trajectories with known states.

    Calendar age is intentionally not aligned across series. Some series include
    dormancy and reactivation so a simple normalized date cannot recover LADT.
    """
    rng = np.random.default_rng(cfg.seed + 111)
    rows: list[dict] = []
    state_order = ["intro", "growth", "mature", "decline", "dormant"]
    for sid in range(cfg.n_series):
        durations = np.array(
            [rng.integers(8, 22), rng.integers(14, 32), rng.integers(22, 52), rng.integers(14, 34), rng.integers(8, 24)],
            dtype=int,
        )
        scale = max(1.0, cfg.history_days / durations.sum())
        durations = np.maximum(5, np.floor(durations * scale).astype(int))
        durations[-1] += cfg.history_days - durations.sum()
        durations[-1] = max(5, durations[-1])
        while durations.sum() > cfg.history_days:
            idx = int(np.argmax(durations))
            durations[idx] -= 1
        while durations.sum() < cfg.history_days:
            durations[2] += 1
        reactivation = sid % 4 == 0
        base = rng.uniform(8, 28)
        peak = base * rng.uniform(1.4, 2.4)
        price0 = rng.uniform(3, 11)
        cursor = 0
        for state_idx, (state, duration) in enumerate(zip(state_order, durations)):
            for within in range(int(duration)):
                t = cursor + within
                p = within / max(1, duration - 1)
                if state == "intro":
                    level = base * (0.15 + 0.35 * p)
                elif state == "growth":
                    level = base * 0.5 + (peak - base * 0.5) * p
                elif state == "mature":
                    level = peak * (1 + 0.04 * np.sin(2 * np.pi * p))
                elif state == "decline":
                    level = peak * (1 - 0.72 * p)
                else:
                    level = peak * (0.08 if not reactivation else 0.08 + 0.42 * max(0, p - 0.45) / 0.55)
                dow = t % 7
                promo = int((t + sid) % 23 == 0)
                seasonal = 1 + 0.16 * np.sin(2 * np.pi * dow / 7)
                latent = max(0.0, level * seasonal + promo * base * 0.25 + rng.normal(0, max(0.6, level * 0.08)))
                stockout = int(rng.random() < min(0.35, 0.04 + 0.12 * promo + 0.1 * (latent > peak)))
                sales = latent * rng.uniform(0.45, 0.82) if stockout else latent
                true_state = "reactivation" if state == "dormant" and reactivation and p > 0.45 else state
                state_base = {"intro": 0.0, "growth": 0.18, "mature": 0.42, "decline": 0.72, "dormant": 0.90, "reactivation": 0.55}[true_state]
                state_width = {"intro": 0.18, "growth": 0.24, "mature": 0.30, "decline": 0.18, "dormant": 0.10, "reactivation": 0.20}[true_state]
                rows.append(
                    {
                        "series_id": f"life_{sid:03d}",
                        "date_idx": t,
                        "sales": float(max(0.0, sales)),
                        "demand": float(max(0.0, latent)),
                        "stockout": stockout,
                        "price": float(price0 * (0.92 if promo else 1.0)),
                        "promo": promo,
                        "category": f"life_cat_{sid % 3}",
                        "store": f"life_store_{sid % 4}",
                        "dow": dow,
                        "true_lifecycle_state": true_state,
                        "true_ladt": float(min(1.0, state_base + state_width * p)),
                    }
                )
            cursor += int(duration)
    return pd.DataFrame(rows).sort_values(["series_id", "date_idx"]).reset_index(drop=True)
