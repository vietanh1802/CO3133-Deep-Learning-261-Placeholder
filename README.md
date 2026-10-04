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

### Assignment 1: Fashion-MNIST, MNIST, CIFAR-10

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

### Assignment 2: SUIM

SUIM is not downloaded automatically and is not in the repository. Unpack it so that
`data/suim/train_val/` and `data/suim/TEST/` exist, each with an `images/` and a `masks/`
folder, then run:

```bash
uv run python -m scripts.eda.run_suim
```

One run produces every number quoted in the A2 dataset proposal. It takes a few minutes,
most of it the near-duplicate search over all 1,635 pairs.

| Output | What it answers |
| --- | --- |
| `results/a2/eda/suim/summary.json` | Pairs on disk, decode and size faults, per-class pixel and image counts for both splits, the resolution histogram, and every duplicate group listed by filename |
| `results/a2/eda/suim/class_distribution.png` | Image count beside pixel share, which rank the classes differently |
| `results/a2/eda/suim/region_area_distribution.png` | How much of a frame a class covers when present, separating rare-and-large from common-and-small |
| `results/a2/eda/suim/resolution_distribution.png` | The eleven resolutions on a log scale |
| `results/a2/splits/suim/train.txt`, `val.txt` | The fixed 1,143 / 287 split, one filename stem per line |

The two split files are committed on purpose. Training reads them rather than re-deriving
the split, so a later change to the code cannot silently move an image between sides.
The statistics live in `src/data/suim.py`; the script only orchestrates and writes.

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
- Checkpoint selection: lowest validation loss (A1); highest validation mIoU (A2)
- Assignment 2 split: fixed in `results/a2/splits/suim/`, seed 42, regenerated by the command above
- Checkpoint reconstruction: run the training commands above

## Documents

- [Assignment 1 draft report](reports/assignment1-draft.md)
- [Assignment 2 M1 dataset proposal](reports/a2_proposal.md)
- [AI Usage Disclosure](AI_USAGE.md)
