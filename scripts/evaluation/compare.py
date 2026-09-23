"""Build the common comparison table and figures for all A1 models."""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from loguru import logger

from src.utils import ExperimentState, setup_logging

TABLE_COLUMNS = (
    "model",
    "runs",
    "accuracy",
    "macro_f1",
    "parameter_count",
    "training_seconds",
    "inference_seconds",
)


def parse_args() -> argparse.Namespace:
    """Parse comparison command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        default="results/a1",
        help="directory containing one subdirectory per evaluated run",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="which split's result files to aggregate",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="where to write the comparison table (default: --results-dir)",
    )
    return parser.parse_args()


def load_records(results_dir: Path, split: str) -> list[dict[str, Any]]:
    """Load every stored evaluation record written by ``evaluate.py``."""
    records: list[dict[str, Any]] = []
    for path in sorted(results_dir.glob(f"*/{split}_result.json")):
        records.append(json.loads(path.read_text(encoding="utf-8")))
    return records


def _mean_std(values: list[float]) -> tuple[float, float]:
    """Return the mean and sample standard deviation of a list of values.

    A single run has no spread and reports a deviation of ``0.0``; the run count
    in the table is what distinguishes it from a genuinely stable result.
    """
    if len(values) == 1:
        return values[0], 0.0
    return statistics.fmean(values), statistics.stdev(values)


def aggregate(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group records by architecture and summarise each group across seeds."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["model"]].append(record)

    rows: list[dict[str, Any]] = []
    for model, model_records in sorted(grouped.items()):
        accuracies = [record["metrics"]["accuracy"] for record in model_records]
        macro_f1s = [record["metrics"]["macro_f1"] for record in model_records]
        inference = [record["inference_seconds"] for record in model_records]
        training = [
            record["training_seconds"]
            for record in model_records
            if record.get("training_seconds") is not None
        ]

        parameter_counts = {record["parameter_count"] for record in model_records}
        if len(parameter_counts) > 1:
            raise ValueError(
                f"model {model!r} has inconsistent parameter counts: {sorted(parameter_counts)}"
            )

        seeds = [record["seed"] for record in model_records]
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"model {model!r} has duplicate seeds: {sorted(seeds)}")

        accuracy_mean, accuracy_std = _mean_std(accuracies)
        macro_f1_mean, macro_f1_std = _mean_std(macro_f1s)
        inference_mean, inference_std = _mean_std(inference)
        training_mean, training_std = _mean_std(training) if training else (None, None)

        rows.append(
            {
                "model": model,
                "runs": len(model_records),
                "seeds": sorted(seeds),
                "accuracy_mean": accuracy_mean,
                "accuracy_std": accuracy_std,
                "macro_f1_mean": macro_f1_mean,
                "macro_f1_std": macro_f1_std,
                "parameter_count": parameter_counts.pop(),
                "training_seconds_mean": training_mean,
                "training_seconds_std": training_std,
                "inference_seconds_mean": inference_mean,
                "inference_seconds_std": inference_std,
                "git_commits": sorted({record["git_commit"] for record in model_records}),
            }
        )
    return rows


def _format_measurement(mean: float | None, deviation: float | None, digits: int) -> str:
    """Render one mean plus or minus deviation cell."""
    if mean is None:
        return "n/a"
    if not deviation:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {deviation:.{digits}f}"


def to_markdown(rows: list[dict[str, Any]]) -> str:
    """Render the comparison rows as a Markdown table for the report and page."""
    header = "| " + " | ".join(TABLE_COLUMNS) + " |"
    separator = "| " + " | ".join("---" for _ in TABLE_COLUMNS) + " |"
    lines = [header, separator]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                (
                    str(row["model"]),
                    str(row["runs"]),
                    _format_measurement(row["accuracy_mean"], row["accuracy_std"], 4),
                    _format_measurement(row["macro_f1_mean"], row["macro_f1_std"], 4),
                    f"{row['parameter_count']:,}",
                    _format_measurement(
                        row["training_seconds_mean"], row["training_seconds_std"], 1
                    ),
                    _format_measurement(
                        row["inference_seconds_mean"], row["inference_seconds_std"], 3
                    ),
                )
            )
            + " |"
        )
    return "\n".join(lines)


def to_csv(rows: list[dict[str, Any]]) -> str:
    """Render the full aggregate with mean and deviation in separate columns."""
    fields = [
        "model",
        "runs",
        "seeds",
        "accuracy_mean",
        "accuracy_std",
        "macro_f1_mean",
        "macro_f1_std",
        "parameter_count",
        "training_seconds_mean",
        "training_seconds_std",
        "inference_seconds_mean",
        "inference_seconds_std",
        "git_commits",
    ]
    lines = [",".join(fields)]
    for row in rows:
        values = []
        for field in fields:
            value = row[field]
            if isinstance(value, list):
                values.append(" ".join(str(item) for item in value))
            elif value is None:
                values.append("")
            else:
                values.append(str(value))
        lines.append(",".join(values))
    return "\n".join(lines)


def main() -> None:
    """Wire stored run results into assignment comparison outputs."""
    args = parse_args()
    setup_logging()
    logger.info(ExperimentState.INITIALIZING)

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(results_dir, args.split)
    if not records:
        raise FileNotFoundError(
            f"no {args.split}_result.json files found under {results_dir}; "
            "run scripts/evaluation/evaluate.py first"
        )
    logger.info("Loaded {} run record(s)", len(records))

    datasets = {record["dataset"] for record in records}
    if len(datasets) > 1:
        raise ValueError(f"records mix datasets and are not comparable: {sorted(datasets)}")
    splits = {record["split"] for record in records}
    if len(splits) > 1:
        raise ValueError(f"records mix splits and are not comparable: {sorted(splits)}")

    rows = aggregate(records)

    markdown = to_markdown(rows)
    markdown_path = output_dir / f"comparison_{args.split}.md"
    markdown_path.write_text(markdown + "\n", encoding="utf-8")

    csv_path = output_dir / f"comparison_{args.split}.csv"
    csv_path.write_text(to_csv(rows) + "\n", encoding="utf-8")

    logger.info(ExperimentState.SAVING)
    logger.info("Wrote {}", markdown_path)
    logger.info("Wrote {}", csv_path)
    print(markdown)

    single_run_models = [row["model"] for row in rows if row["runs"] == 1]
    if single_run_models:
        logger.warning(
            "Single-seed models (no spread reported): {}",
            ", ".join(single_run_models),
        )
    logger.success(ExperimentState.COMPLETED)


if __name__ == "__main__":
    main()
