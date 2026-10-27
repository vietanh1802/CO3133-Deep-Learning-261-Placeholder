"""Rebuild saved result figures without training or evaluating models.

Run from the repository root: python -m scripts.evaluation.replot
"""

import hashlib
import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST

from src.data import build_dataloaders, class_names
from src.eda.report import run_dataset_eda
from src.evaluation.classification import evaluate_model
from src.models import ImageClassifier, build_model
from src.plots.classification import (
    plot_comparative_learning_curves,
    plot_learning_curves,
    plot_prediction_examples,
    plot_shared_predictions,
)
from src.plots.common import plot_confusion_matrix
from src.plots.segmentation import plot_segmentation_curves
from src.training.checkpoint import load_checkpoint


def main():
    """Replace plots while preserving every saved result and sample selection."""
    results = Path("results")
    evidence_files = [p for p in results.rglob("*") if p.suffix in {".json", ".pt", ".csv"}]
    before = {p: hashlib.sha256(p.read_bytes()).digest() for p in evidence_files}
    existing = set(results.rglob("*.png"))
    datasets = {}
    histories = {}
    for path in sorted((results / "a1").glob("*/training_summary.json")):
        directory = path.parent
        summary = json.loads(path.read_text())
        plot_learning_curves(
            summary["train_loss"], summary["validation_loss"], directory / "learning_curves.png"
        )
        histories[directory.name] = summary
        record = json.loads((directory / "test_result.json").read_text())
        name = record["dataset"]
        if name not in datasets:
            cls = {"mnist": MNIST, "fashion_mnist": FashionMNIST, "cifar10": CIFAR10}[name]
            datasets[name] = cls(root="data", train=False, download=False)
        dataset = datasets[name]
        names = class_names(name)
        errors = json.loads((directory / "test_error_analysis.json").read_text())
        stored = directory / "test_evidence.pt"
        if stored.exists():
            evidence = torch.load(stored, map_location="cpu", weights_only=True)
            matrix = evidence["confusion_matrix"].numpy()
        else:
            # Preliminary runs lack serialized predictions; recover counts from their checkpoint.
            config = record["config"]
            model = ImageClassifier(build_model(record["model"], config["model_config"]))
            load_checkpoint(record["checkpoint"]["path"], model, restore_rng=False)
            loaders = build_dataloaders(
                name,
                {
                    key: config[key]
                    for key in ("data_dir", "batch_size", "val_fraction", "seed", "num_workers")
                },
            )
            matrix = evaluate_model(
                model, loaders.test, torch.device("cpu")
            ).metrics.confusion_matrix.numpy()
        assert np.isclose(np.trace(matrix) / matrix.sum(), record["metrics"]["accuracy"])
        plot_confusion_matrix(matrix, names, directory / "test_confusion_matrix.png")
        for key, title in [
            ("confident_correct", "Highest-confidence correct predictions"),
            ("confident_errors", "Confident errors"),
            ("uncertain_errors", "Uncertain errors"),
        ]:
            examples = errors[key]
            images = []
            for example in examples:
                image = np.asarray(dataset.data[example["sample_index"]])
                images.append(image[None] if image.ndim == 2 else image.transpose(2, 0, 1))
                assert int(dataset.targets[example["sample_index"]]) == example["target"]
            plot_prediction_examples(
                images,
                [e["target"] for e in examples],
                [e["prediction"] for e in examples],
                [e["confidence"] for e in examples],
                names,
                directory / f"test_{key}.png",
                title=title,
            )
        print(directory, flush=True)
    selection = json.loads((results / "a1/comparison_selection.json").read_text())
    loss_max = max(
        max(s[k])
        for s in histories.values()
        if s["epochs"] == 50
        for k in ("train_loss", "validation_loss")
    )
    for name, selected in selection.items():
        group = {
            model: histories[f"{name}_{model.lower()}_seed42"]
            for model in ("Linear", "MLP", "CNN", "GRU", "Transformer")
        }
        plot_comparative_learning_curves(
            group, results / f"a1/comparison/{name}_learning.png", loss_max=loss_max
        )
        samples = selected["samples"]
        images = [np.asarray(datasets[name].data[s["sample_index"]]) for s in samples]
        plot_shared_predictions(
            images, samples, class_names(name), results / f"a1/comparison/{name}_shared_images.png"
        )
        for model in (selected["baseline"], "cnn"):
            evidence = torch.load(
                results / f"a1/{name}_{model}_seed42/test_evidence.pt",
                map_location="cpu",
                weights_only=True,
            )
            plot_confusion_matrix(
                evidence["confusion_matrix"].numpy(),
                class_names(name),
                results / f"a1/comparison/{name}_{model}_confusion.png",
            )
    for directory in (results / "a1/eda").iterdir():
        if directory.is_dir():
            name = directory.name
            cls = {"mnist": MNIST, "fashion_mnist": FashionMNIST, "cifar10": CIFAR10}[name]
            with TemporaryDirectory() as temporary:
                run_dataset_eda(
                    train_dataset=cls(root="data", train=True, download=False),
                    test_dataset=cls(root="data", train=False, download=False),
                    class_names=class_names(name),
                    dataset_name=name,
                    display_name={
                        "fashion_mnist": "Fashion-MNIST",
                        "mnist": "MNIST",
                        "cifar10": "CIFAR-10",
                    }[name],
                    output_dir=temporary,
                )
                for figure in Path(temporary).iterdir():
                    if figure.suffix in {".png", ".pdf"}:
                        shutil.copyfile(figure, directory / figure.name)
    for path in (results / "a2").glob("*/training_summary.json"):
        history = json.loads(path.read_text())["history"]
        plot_segmentation_curves(
            train_loss=[epoch["train_loss"] for epoch in history],
            validation_loss=[epoch["validation_loss"] for epoch in history],
            validation_miou=[epoch["validation_miou"] for epoch in history],
            output_path=path.parent / "learning_curves.png",
        )
    for p, digest in before.items():
        assert hashlib.sha256(p.read_bytes()).digest() == digest, f"Saved evidence changed: {p}"
    # Replace existing website and report copies by their result-relative paths.
    for root in (Path("web/images"), Path("reports")):
        for target in root.rglob("*"):
            if target.suffix not in {".png", ".pdf"}:
                continue
            candidates = [
                p
                for p in results.rglob(target.name)
                if tuple(p.parts[-min(2, len(target.parts)) :])
                == tuple(target.parts[-min(2, len(target.parts)) :])
            ]
            if "comparison" in target.parts:
                source = results / "a1/comparison" / target.name
                candidates = [source] if source.exists() else []
            if len(candidates) == 1:
                shutil.copyfile(candidates[0], target)
    print(
        f"Regenerated classification plots and segmentation curves. Original PNGs: {len(existing)}."
    )
    print("SUIM EDA requires the absent raw dataset; its three plots were not regenerated.")


if __name__ == "__main__":
    main()
