"""Compare saved classification runs without mixing datasets or training budgets."""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST

from src.data import build_dataloaders, class_names
from src.evaluation.classification_metrics import compute_classification_metrics
from src.plots.classification import plot_comparative_learning_curves, plot_shared_predictions
from src.plots.common import plot_confusion_matrix
from src.utils import ExperimentState, setup_logging

TABLE_COLUMNS = (
    "dataset",
    "epochs",
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
    parser.add_argument("--epochs", type=int, help="include only this training budget")
    parser.add_argument(
        "--figures-dir", type=Path, help="export comparison figures and sample indices"
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
    """Summarise seeds within each dataset, training budget, and architecture."""
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["dataset"], record["config"]["epochs"], record["model"])].append(record)

    rows: list[dict[str, Any]] = []
    for (dataset, epochs, model), model_records in sorted(grouped.items()):
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
                "dataset": dataset,
                "epochs": epochs,
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
                    str(row["dataset"]),
                    str(row["epochs"]),
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
        "dataset",
        "epochs",
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


def select_examples(targets, baseline, cnn):
    """Take the first two official test indices per disagreement category."""
    baseline_correct, cnn_correct = baseline == targets, cnn == targets
    return {
        "CNN corrects baseline": np.flatnonzero(~baseline_correct & cnn_correct)[:2],
        "Both miss": np.flatnonzero(~baseline_correct & ~cnn_correct)[:2],
        "Baseline correct, CNN wrong": np.flatnonzero(baseline_correct & ~cnn_correct)[:2],
    }


def make_figures(records, results_dir, output_dir):
    """Validate local test evidence and regenerate curves and paired error figures."""
    if any(r["split"] != "test" for r in records):
        raise ValueError("paired examples require the official unshuffled test split")
    groups = defaultdict(dict)
    histories = {}
    evidence = {}
    labels = {
        "linear": "Linear",
        "mlp": "MLP",
        "cnn": "CNN",
        "gru": "GRU",
        "transformer": "Transformer",
    }
    for record in records:
        dataset, model, name = record["dataset"], record["model"], record["name"]
        if model in groups[dataset]:
            raise ValueError("figures require one run per dataset/model; select a training budget")
        groups[dataset][model] = record
        directory = results_dir / name
        summary = json.loads((directory / "training_summary.json").read_text())
        stored = torch.load(directory / "test_evidence.pt", map_location="cpu", weights_only=True)
        assert summary["name"] == name and summary["model"] == model
        assert summary["dataset"] == dataset
        assert summary["config"] == record["config"] == record["checkpoint"]["config"]
        assert summary["best_epoch"] == record["checkpoint"]["epoch"]
        assert np.isclose(
            summary["best_validation_loss"], record["checkpoint"]["best_validation_loss"]
        )
        assert len(summary["train_loss"]) == len(summary["validation_loss"]) == summary["epochs"]
        assert np.isfinite([summary["train_loss"], summary["validation_loss"]]).all()
        assert np.isclose(min(summary["validation_loss"]), summary["best_validation_loss"])
        assert torch.equal(stored["sample_indices"], torch.arange(record["sample_count"]))
        assert torch.isfinite(stored["probabilities"]).all()
        assert torch.equal(stored["predictions"], stored["probabilities"].argmax(1))
        metrics = compute_classification_metrics(
            stored["targets"], stored["predictions"], num_classes=10
        )
        assert np.isclose(metrics.accuracy, record["metrics"]["accuracy"])
        assert np.isclose(metrics.macro_f1, record["metrics"]["macro_f1"])
        assert torch.equal(metrics.confusion_matrix, stored["confusion_matrix"])
        histories[name], evidence[name] = summary, stored
    for group in groups.values():
        if len({(r["seed"], r["config"]["epochs"]) for r in group.values()}) != 1:
            raise ValueError("comparison figures require matching seeds and training budgets")
        if not {"linear", "mlp", "cnn"} <= group.keys():
            raise ValueError("paired comparison requires Linear, MLP, and CNN")
    output_dir.mkdir(parents=True, exist_ok=True)
    loss_max = max(
        max(s[key]) for s in histories.values() for key in ("train_loss", "validation_loss")
    )
    selections = {}
    for dataset, group in sorted(groups.items()):
        ordered = [m for m in labels if m in group]
        plot_comparative_learning_curves(
            {labels[m]: histories[group[m]["name"]] for m in ordered},
            output_dir / f"{dataset}_learning.png",
            loss_max=loss_max,
        )
        baseline = min(
            ("linear", "mlp"), key=lambda m: histories[group[m]["name"]]["best_validation_loss"]
        )
        config = group[baseline]["config"]
        loaders = build_dataloaders(
            dataset,
            {
                key: config[key]
                for key in ("data_dir", "batch_size", "val_fraction", "seed", "num_workers")
            },
        )
        test = loaders.test.dataset
        assert isinstance(test, (MNIST, FashionMNIST, CIFAR10))
        targets = evidence[group[baseline]["name"]]["targets"].numpy()
        assert np.array_equal(targets, np.asarray(test.targets))
        for record in group.values():
            assert np.array_equal(targets, evidence[record["name"]]["targets"].numpy())
        names = class_names(dataset)
        for model in (baseline, "cnn"):
            plot_confusion_matrix(
                evidence[group[model]["name"]]["confusion_matrix"].numpy(),
                names,
                output_dir / f"{dataset}_{model}_confusion.png",
            )
        categories = select_examples(
            targets,
            evidence[group[baseline]["name"]]["predictions"].numpy(),
            evidence[group["cnn"]["name"]]["predictions"].numpy(),
        )
        samples: list[dict[str, Any]] = []
        for category, indices in categories.items():
            for index in indices.tolist():
                samples.append(
                    {
                        "category": category,
                        "sample_index": index,
                        "target": int(targets[index]),
                        "predictions": {
                            m: int(evidence[group[m]["name"]]["predictions"][index])
                            for m in ordered
                        },
                    }
                )
        display_samples = [
            {**s, "predictions": {labels[m]: p for m, p in s["predictions"].items()}}
            for s in samples
        ]
        plot_shared_predictions(
            [test.data[s["sample_index"]] for s in samples],
            display_samples,
            names,
            output_dir / f"{dataset}_shared_images.png",
        )
        selections[dataset] = {
            "baseline": baseline,
            "rule": "lowest validation loss among Linear and MLP; ties favor Linear",
            "categories": {key: indices.tolist() for key, indices in categories.items()},
            "samples": samples,
        }
    (results_dir / "comparison_selection.json").write_text(json.dumps(selections, indent=2) + "\n")


def main() -> None:
    """Wire stored run results into assignment comparison outputs."""
    args = parse_args()
    setup_logging()
    logger.info(ExperimentState.INITIALIZING)

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(results_dir, args.split)
    if args.epochs is not None:
        records = [r for r in records if r["config"]["epochs"] == args.epochs]
    if not records:
        raise FileNotFoundError(
            f"no {args.split}_result.json files found under {results_dir}; "
            "run scripts/evaluation/evaluate.py first"
        )
    logger.info("Loaded {} run record(s)", len(records))

    splits = {record["split"] for record in records}
    if len(splits) > 1:
        raise ValueError(f"records mix splits and are not comparable: {sorted(splits)}")

    rows = aggregate(records)
    if args.figures_dir:
        make_figures(records, results_dir, args.figures_dir)

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
