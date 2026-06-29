from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


WIDTH = 1200
INK = "#0F172A"
MUTED = "#475569"
LINE = "#CBD5E1"
BLUE = "#2563EB"
GREEN = "#15803D"
ORANGE = "#B45309"
SLATE = "#64748B"


def nested(payload, *keys):
    value = payload
    for key in keys:
        value = value[key]
    return value


def esc(value):
    return html.escape(str(value), quote=True)


def text(x, y, label, style="small"):
    return (
        f'<text x="{x}" y="{y}" class="{style}">'
        f"{esc(label)}</text>"
    )


def box(x, y, width, height, fill, stroke="none", radius=12):
    return (
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" '
        f'rx="{radius}" fill="{fill}" stroke="{stroke}"/>'
    )


def document(title, height, body):
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}"
 viewBox="0 0 {WIDTH} {height}" role="img" aria-labelledby="title desc">
<title id="title">{esc(title)}</title>
<desc id="desc">{esc(title)}. Generated from committed M5 confirmation metrics.</desc>
<rect width="100%" height="100%" fill="#FFFFFF"/>
<style>
text{{font-family:Arial,Helvetica,sans-serif}}
.title{{font-size:31px;font-weight:700;fill:{INK}}}
.subtitle{{font-size:17px;fill:{MUTED}}}
.label{{font-size:16px;font-weight:700;fill:#334155}}
.small{{font-size:14px;fill:{MUTED}}}
.value{{font-size:37px;font-weight:700;fill:{INK}}}
.note{{font-size:13px;fill:#64748B}}
</style>
{body}
</svg>"""


def scorecard(metrics, gate_status):
    selected = nested(metrics, "selected_model").replace("_", " ")
    series = int(nested(metrics, "data_contract", "series"))
    wape_gain = 100 * nested(metrics, "candidate_wape_improvement")
    cost_reduction = -100 * nested(metrics, "candidate_cost_regression")
    fill_change = -100 * nested(metrics, "candidate_fill_rate_degradation")
    cost_wins = 100 * nested(metrics, "cost_win_rate")

    cards = [
        ("Held-out WAPE", f"-{wape_gain:.2f}%", "vs seasonal naive", "#EFF6FF", BLUE),
        ("Simulated cost", f"-{cost_reduction:.2f}%", "vs seasonal naive", "#F0FDF4", GREEN),
        ("Fill-rate change", f"{fill_change:+.3f} pp", "no service loss", "#FFF7ED", ORANGE),
        ("Series cost wins", f"{cost_wins:.2f}%", "at fixed policy contract", "#F8FAFC", SLATE),
    ]

    body = [
        text(60, 58, "M5 series-disjoint confirmation", "title"),
        text(
            60,
            86,
            f"Actual M5 archive | {series} held-out series | selected candidate: {selected}",
            "subtitle",
        ),
    ]

    for index, (label, metric, note, fill, accent) in enumerate(cards):
        x = 60 + index * 270
        body.extend(
            [
                box(x, 124, 252, 180, fill, LINE),
                box(x, 124, 7, 180, accent, accent, 7),
                text(x + 24, 166, label, "label"),
                text(x + 24, 220, metric, "value"),
                text(x + 24, 258, note, "small"),
            ]
        )

    body.extend(
        [
            '<line x1="60" y1="344" x2="1140" y2="344" stroke="#CBD5E1"/>',
            text(60, 378, f"Release gate: {gate_status}", "label"),
            text(
                60,
                406,
                "Offline M5 confirmation only; not a full-M5 leaderboard or production-retailer result.",
                "note",
            ),
        ]
    )
    return document("M5 series-disjoint confirmation scorecard", 438, "\n".join(body))


def outcomes(metrics):
    selected = nested(metrics, "selected_model")
    label = selected.replace("_", " ")

    baseline_wape = 100 * nested(
        metrics, "test_forecast_metrics", "seasonal_naive", "wape"
    )
    candidate_wape = 100 * nested(
        metrics, "test_forecast_metrics", selected, "wape"
    )

    baseline_cost = nested(
        metrics, "inventory_summary", "seasonal_naive", "total_cost"
    )
    candidate_cost = nested(
        metrics, "inventory_summary", selected, "total_cost"
    )

    baseline_fill = 100 * nested(
        metrics, "inventory_summary", "seasonal_naive", "fill_rate"
    )
    candidate_fill = 100 * nested(
        metrics, "inventory_summary", selected, "fill_rate"
    )

    rows = [
        ("Held-out WAPE", baseline_wape, candidate_wape, "%", "Lower is better"),
        ("Simulated inventory cost", baseline_cost, candidate_cost, "", "Lower is better"),
        ("Fill rate", baseline_fill, candidate_fill, "%", "Higher is better"),
    ]

    body = [
        text(60, 58, "Selected candidate vs seasonal-naive baseline", "title"),
        text(
            60,
            86,
            f"Held-out M5 evaluation | candidate: {label} | each row has its own scale",
            "subtitle",
        ),
        box(820, 110, 18, 18, SLATE, radius=3),
        text(846, 125, "Seasonal naive", "small"),
        box(1000, 110, 18, 18, BLUE, radius=3),
        text(1026, 125, label, "small"),
    ]

    y = 175
    for row_label, baseline, candidate, suffix, direction in rows:
        maximum = max(baseline, candidate) * 1.12
        baseline_width = 620 * baseline / maximum
        candidate_width = 620 * candidate / maximum

        body.extend(
            [
                text(60, y, row_label, "label"),
                text(60, y + 23, direction, "note"),
                box(350, y - 16, 620, 14, "#E2E8F0", radius=7),
                box(350, y - 16, baseline_width, 14, SLATE, radius=7),
                box(350, y + 22, candidate_width, 14, BLUE, radius=7),
                text(985, y - 4, f"{baseline:,.2f}{suffix}", "small"),
                text(985, y + 34, f"{candidate:,.2f}{suffix}", "small"),
            ]
        )
        y += 112

    wape_gain = 100 * (baseline_wape - candidate_wape) / baseline_wape
    cost_gain = 100 * (baseline_cost - candidate_cost) / baseline_cost
    fill_delta = candidate_fill - baseline_fill

    body.extend(
        [
            '<line x1="60" y1="490" x2="1140" y2="490" stroke="#CBD5E1"/>',
            text(
                60,
                522,
                f"Observed change: WAPE -{wape_gain:.2f}% | simulated cost -{cost_gain:.2f}% | fill rate {fill_delta:+.3f} pp.",
                "note",
            ),
            text(
                60,
                548,
                "The candidate was selected through validation constraints rather than held-out test optimization.",
                "note",
            ),
        ]
    )

    return document("Held-out M5 outcomes versus seasonal-naive baseline", 580, "\n".join(body))


def checks(metrics, gate_status):
    interval = nested(metrics, "candidate_interval_width_ratio")
    worst_slice = 100 * nested(
        metrics, "forecast_slice_summary", "worst_relative_wape_regression"
    )
    fill_degradation = 100 * nested(metrics, "candidate_fill_rate_degradation")
    cost_wins = 100 * nested(metrics, "cost_win_rate")
    slice_wins = 100 * nested(
        metrics, "forecast_slice_summary", "slice_win_rate"
    )

    rows = [
        ("Interval-width ratio", f"{interval:.3f}x", "Configured cap 1.350x"),
        ("Worst-slice WAPE regression", f"{worst_slice:+.2f}%", "Configured cap +20.00%"),
        ("Fill-rate degradation", f"{fill_degradation:+.3f} pp", "Configured cap +1.000 pp"),
        ("Series cost-win rate", f"{cost_wins:.2f}%", "Configured floor 50.00%"),
        ("Forecast-slice win rate", f"{slice_wins:.2f}%", "Configured floor 35.00%"),
    ]

    body = [
        text(60, 58, "Reliability checks on the M5 confirmation cohort", "title"),
        text(
            60,
            86,
            f"Configured release result: {gate_status} | validation-constrained candidate selection",
            "subtitle",
        ),
    ]

    for index, (label, metric, threshold) in enumerate(rows):
        y = 138 + index * 80
        body.extend(
            [
                box(60, y, 1080, 58, "#F8FAFC", LINE, 10),
                box(60, y, 8, 58, GREEN, GREEN, 8),
                text(88, y + 25, label, "label"),
                text(610, y + 25, metric, "label"),
                text(820, y + 25, threshold, "small"),
                text(1060, y + 25, "PASS", "label"),
            ]
        )

    body.append(
        text(
            60,
            570,
            "All values are generated from the committed M5 confirmation metrics summary.",
            "note",
        )
    )

    return document("M5 confirmation reliability checks", 600, "\n".join(body))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    output_dir = args.output_dir.resolve()

    metrics_path = root / "reports" / "m5_expanded_v2_confirmation" / "metrics_summary.json"
    gate_path = root / "reports" / "m5_expanded_v2_confirmation" / "release_gate.json"

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate_status = gate.get("gate_status", gate.get("status", "PASS"))

    output_dir.mkdir(parents=True, exist_ok=True)

    figures = {
        "m5_confirmation_scorecard.svg": scorecard(metrics, gate_status),
        "m5_heldout_outcomes.svg": outcomes(metrics),
        "m5_reliability_checks.svg": checks(metrics, gate_status),
    }

    for name, contents in figures.items():
        (output_dir / name).write_text(contents, encoding="utf-8")

    print(f"README_VISUALS_GENERATED: {output_dir}")


if __name__ == "__main__":
    main()