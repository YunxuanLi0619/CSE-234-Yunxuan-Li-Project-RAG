#!/usr/bin/env python3
"""Generate RapidFire config comparison report artifacts from logged metrics."""

from __future__ import annotations

import csv
import html
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = ROOT / "logs"
REPORTS_DIR = ROOT / "reports"
METRICS_PATH = LOGS_DIR / "rapidfire_project1_metrics.csv"
LOG_PATH = LOGS_DIR / "rapidfire_project1_rapidfire.log"
MARKDOWN_PATH = REPORTS_DIR / "rapidfire_config_comparison.md"
SVG_PATH = REPORTS_DIR / "rapidfire_config_comparison.svg"


METRIC_COLUMNS = [
    ("source_f1_at_5", "Source F1@5", "#2563eb"),
    ("source_precision_at_5", "Precision@5", "#16a34a"),
    ("source_recall_at_5", "Recall@5", "#f97316"),
    ("reference_token_recall", "Ref token recall", "#7c3aed"),
]


def short_embedding(model: str) -> str:
    if "bge-small" in model:
        return "BGE-small"
    if "MiniLM" in model:
        return "MiniLM-L6"
    return model.rsplit("/", 1)[-1]


def short_reranker(model: str) -> str:
    return "none" if model == "none" else "cross-encoder"


def read_metrics() -> list[dict[str, str]]:
    with METRICS_PATH.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def find_log_evidence() -> list[str]:
    needles = [
        "Using existing dispatcher/api server",
        "Running multi-config experiment with 8 shard(s)",
        "Received 8 pipeline configuration(s)",
        "Building document index",
        "Document index built successfully",
    ]
    evidence: list[str] = []
    if not LOG_PATH.exists():
        return evidence
    with LOG_PATH.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if any(needle in line for needle in needles):
                evidence.append(line.strip())
            if len(evidence) >= 8:
                break
    return evidence


def config_label(row: dict[str, str]) -> str:
    search = row["search_type"]
    rerank = short_reranker(row["reranker_model"])
    prompt = "structured" if row["run_key"] in {"2", "4", "5", "7"} else "concise"
    return (
        f"Run {row['run_key']}: {short_embedding(row['embedding_model'])}, "
        f"{row['chunk_size']}/{row['chunk_overlap']}, {search}, {rerank}, {prompt}"
    )


def write_markdown(rows: list[dict[str, str]]) -> None:
    best_f1 = max(rows, key=lambda row: float(row["source_f1_at_5"]))
    best_ref = max(rows, key=lambda row: float(row["reference_token_recall"]))
    evidence = find_log_evidence()

    lines = [
        "# RapidFire RAG Config Comparison",
        "",
        "This artifact summarizes the RapidFire AI OSS multi-config experiment logs and metrics.",
        "It is generated from `logs/rapidfire_project1_metrics.csv` and links the comparison chart below.",
        "",
        "![RapidFire config comparison](rapidfire_config_comparison.svg)",
        "",
        "## Launch and Run Evidence",
        "",
        "- Source log: `logs/rapidfire_project1_rapidfire.log`",
        "- Metrics table: `logs/rapidfire_project1_metrics.csv`",
        "- Raw results: `logs/rapidfire_project1_results.json`",
        "- Runs info: `logs/rapidfire_project1_runs_info.csv`",
        "",
    ]

    if evidence:
        lines.append("Selected log lines showing RapidFire startup and multi-config execution:")
        lines.append("")
        for item in evidence:
            lines.append(f"- `{item}`")
        lines.append("")

    lines.extend(
        [
            "## Summary",
            "",
            (
                f"- Best Source F1@5: Run {best_f1['run_key']} "
                f"({config_label(best_f1)}) = {float(best_f1['source_f1_at_5']):.4f}."
            ),
            (
                f"- Best reference-token recall: Run {best_ref['run_key']} "
                f"({config_label(best_ref)}) = {float(best_ref['reference_token_recall']):.4f}."
            ),
            "- All runs used `api-llama-4-scout` with temperature 0.0 and max_tokens 300.",
            "",
            "## Config and Metric Table",
            "",
            "| Run | Embedding | Chunk | Search | Reranker | Samples | Precision@5 | Recall@5 | Source F1@5 | Hit Rate | Ref Token Recall |",
            "|---:|---|---:|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for row in rows:
        lines.append(
            "| {run_key} | {embedding} | {chunk_size}/{chunk_overlap} | {search_type} | {reranker} | "
            "{samples_processed} | {precision:.4f} | {recall:.4f} | {f1:.4f} | {hit:.4f} | {ref:.4f} |".format(
                run_key=row["run_key"],
                embedding=short_embedding(row["embedding_model"]),
                chunk_size=row["chunk_size"],
                chunk_overlap=row["chunk_overlap"],
                search_type=row["search_type"],
                reranker=short_reranker(row["reranker_model"]),
                samples_processed=row["samples_processed"],
                precision=float(row["source_precision_at_5"]),
                recall=float(row["source_recall_at_5"]),
                f1=float(row["source_f1_at_5"]),
                hit=float(row["source_hit_rate"]),
                ref=float(row["reference_token_recall"]),
            )
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "Run 5, the BGE-small 256/32 similarity-search configuration without reranking, "
                "had the strongest Source F1@5. Run 7 had the strongest reference-token recall, "
                "but its source F1 was slightly lower. In this corpus, BGE improved recall over "
                "MiniLM, while MMR and cross-encoder reranking did not consistently improve source-span metrics."
            ),
            "",
        ]
    )

    MARKDOWN_PATH.write_text("\n".join(lines), encoding="utf-8")


def svg_text(x: float, y: float, text: str, size: int = 13, weight: str = "400", anchor: str = "start") -> str:
    safe = html.escape(text)
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="#111827">{safe}</text>'
    )


def write_svg(rows: list[dict[str, str]]) -> None:
    width = 1180
    height = 730
    left = 390
    top = 92
    row_h = 68
    chart_w = 620
    axis_x = left
    axis_y = top + len(rows) * row_h + 18

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        svg_text(32, 38, "RapidFire RAG Config Comparison", 24, "700"),
        svg_text(32, 64, "Metrics from logs/rapidfire_project1_metrics.csv; each run processed 24 validation samples.", 13),
    ]

    legend_x = left
    for index, (_, label, color) in enumerate(METRIC_COLUMNS):
        x = legend_x + index * 150
        parts.append(f'<rect x="{x}" y="36" width="14" height="14" rx="2" fill="{color}"/>')
        parts.append(svg_text(x + 20, 48, label, 12))

    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        x = axis_x + tick * chart_w
        parts.append(f'<line x1="{x:.1f}" y1="{top - 10}" x2="{x:.1f}" y2="{axis_y}" stroke="#e5e7eb" stroke-width="1"/>')
        parts.append(svg_text(x, axis_y + 18, f"{tick:.2f}", 11, anchor="middle"))

    parts.append(f'<line x1="{axis_x}" y1="{axis_y}" x2="{axis_x + chart_w}" y2="{axis_y}" stroke="#9ca3af" stroke-width="1"/>')

    for idx, row in enumerate(rows):
        y = top + idx * row_h
        if idx % 2:
            parts.append(f'<rect x="24" y="{y - 20}" width="{width - 48}" height="{row_h - 6}" fill="#f9fafb"/>')
        parts.append(svg_text(32, y, config_label(row), 13, "600"))
        parts.append(svg_text(32, y + 19, f"samples={row['samples_processed']}, time={row['processing_time']}", 11))

        for bar_idx, (column, label, color) in enumerate(METRIC_COLUMNS):
            value = float(row[column])
            bar_y = y - 14 + bar_idx * 14
            bar_h = 9
            bar_w = value * chart_w
            parts.append(f'<rect x="{axis_x}" y="{bar_y:.1f}" width="{chart_w}" height="{bar_h}" rx="2" fill="#eef2f7"/>')
            parts.append(f'<rect x="{axis_x}" y="{bar_y:.1f}" width="{bar_w:.1f}" height="{bar_h}" rx="2" fill="{color}"/>')
            parts.append(svg_text(axis_x + bar_w + 7, bar_y + 8, f"{value:.3f}", 10))

    parts.append(svg_text(axis_x + chart_w / 2, axis_y + 44, "Metric value", 12, "600", "middle"))
    parts.append("</svg>")
    SVG_PATH.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = read_metrics()
    write_markdown(rows)
    write_svg(rows)
    print(f"Wrote {MARKDOWN_PATH.relative_to(ROOT)}")
    print(f"Wrote {SVG_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
