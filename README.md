# CO3133 - Deep Learning and Its Applications (Semester 261)

Course project repository. Group: Placeholder

**Project website:** https://vietanh1802.github.io/CO3133-Deep-Learning-261-Placeholder/

**Assignment pages:** [Assignment 1](web/assignment1.html) · [Assignment 2](web/assignment2.html) · [Assignment 3](web/assignment3.html)

## Installation

```bash
uv sync --dev
uv run pre-commit install
```

## Dataset Preparation

Fashion-MNIST, MNIST, and CIFAR-10 are downloaded and integrity-checked automatically.
Fashion-MNIST remains the Assignment 1 evaluation dataset; MNIST and CIFAR-10 are available
for development and later experiments. Generate the A1 EDA figures with:

```bash
uv run python -m scripts.eda.run_fashion_mnist
uv run python -m scripts.eda.run_mnist
```

Each command writes the following to `results/a1/eda/<dataset>/`:

| Output | What it answers |
| --- | --- |
| `class_distribution.png` | How many samples per class, and how imbalanced the set is |
| `representative_samples.png` | What the raw inputs look like, three per class |
| `split_class_distribution.png` | Whether the train and test splits share the same class mix |
| `pixel_intensity_distribution.png` | How intensities are spread, which justifies the normalization constants |
| `class_mean_images.png` | The average image of each class |
| `class_similarity.png` | Which classes look alike before any training, i.e. the confusions to expect |
| `foreground_coverage.png` | How much of the frame each class occupies |
| `summary.json` | Every number above, machine-readable, for the report to cite |

The analysis itself lives in `src/data/eda.py` (statistics), `src/data/eda_plots.py`
(figures), and `src/data/eda_report.py` (one shared run), so the entry points only have
to name a dataset.

## Training

```bash
uv run python -m scripts.training.train_image_classification --data fashion_mnist --config configs/image_classification/fashion_mnist/linear.yaml
uv run python -m scripts.training.train_image_classification --data fashion_mnist --config configs/image_classification/fashion_mnist/mlp.yaml
```

Development dataset runs use their matching entry points and shape-compatible configs:

```bash
uv run python -m scripts.training.train_image_classification --data mnist --config configs/image_classification/mnist/linear.yaml
uv run python -m scripts.training.train_image_classification --data cifar10 --config configs/image_classification/cifar10/cnn.yaml
```

The best-validation checkpoints are written to `checkpoints/<run-name>/best.pt`. Training
logs, curves, and exact summaries are written to `results/a1/<run-name>/`.

Package optional checkpoint downloads separately for each dataset:

```bash
uv run python -m scripts.release.package_checkpoints --data fashion_mnist
uv run python -m scripts.release.package_checkpoints --data mnist
uv run python -m scripts.release.package_checkpoints --data cifar10
```

Each command writes `artifacts/<dataset>-checkpoints.zip`. Upload these archives as GitHub
Release assets; they are not downloaded when cloning the repository.

## Evaluation

```bash
uv run python -m scripts.evaluation.evaluate --data fashion_mnist --config configs/image_classification/fashion_mnist/linear.yaml
uv run python -m scripts.evaluation.evaluate --data fashion_mnist --config configs/image_classification/fashion_mnist/mlp.yaml
uv run python -m scripts.evaluation.compare --results-dir results/a1
```

Each evaluation writes the required confusion matrix, per-class accuracy and support,
most-confused class pairs, and representative correct, confident-error, and uncertain-error
figures under `results/a1/<run-name>/`.

## Reproducibility

- Configuration files: `configs/image_classification/<dataset>/`
- Draft random seed: 42
- Dependency versions: locked in `uv.lock`
- Draft hardware: Apple Silicon MPS for training; CPU for reported inference timing
- Checkpoint selection: lowest validation loss
- Checkpoint reconstruction: run the training commands above

## Documents

- [Assignment 1 draft report](reports/assignment1-draft.md)
- [AI Usage Disclosure](AI_USAGE.md)
