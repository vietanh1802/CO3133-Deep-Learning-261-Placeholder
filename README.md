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

Each command writes the following to `outputs/a1_v1/results/eda/<dataset>/`:

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

## Output layout

Local and Colab use the same assignment/version bundle:

```text
outputs/a1_v1/                 # Local
MyDrive/BTL_DL_Result/VanhResult/a1_v1/  # Drive
  checkpoints/<run-name>/best.pt
  results/<run-name>/
  results/eda/<dataset>/
  configs/<run-name>.yaml
  configs/<run-name>_test.yaml
  console/<run-name>/train.log
  console/<run-name>/evaluate.log
```

The repository's top-level `configs/image_classification/` contains source configs.
The bundle's `configs/` stores resolved settings used by each command. New A2/A3
pipelines can use separate `outputs/a2_v1/` and `outputs/a3_v1/` bundles; this
change only updates A1.

After pulling this change, move any ignored local checkpoints and logs left at the
old paths by running this once from the repository root (safe to repeat):

```bash
uv run python -m scripts.maintenance.migrate_a1_outputs
```

It moves `checkpoints/` and `results/a1/` into the bundle and relocates logs to
`console/`. Different files at the same destination stop the migration before
any file is moved. Existing JSON/checkpoint metadata keeps its original paths as
a historical record; current commands find artifacts using the new config paths.
Checkpoint weights, evidence tensors, logs, and ZIPs remain ignored by Git.

The [Colab notebook](notebooks/A1_Train_All_Models_CUDA.ipynb) trains and evaluates
six Fashion-MNIST models, shows live progress, and copies the same bundle layout
to Drive without an extra run-id folder. Run it after this layout change is merged
into `a1/m2`. Files with the same names on Drive are overwritten.

## Training

```bash
uv run python -m scripts.training.train_image_classification --data fashion_mnist --config configs/image_classification/fashion_mnist/linear.yaml
uv run python -m scripts.training.train_image_classification --data fashion_mnist --config configs/image_classification/fashion_mnist/mlp.yaml
uv run python -m scripts.training.train_image_classification --data fashion_mnist --config configs/image_classification/fashion_mnist/lstm.yaml
```

The Fashion-MNIST LSTM reads each image as 28 row timesteps with 28 input features,
then passes the final hidden state to the shared classification head. Its config uses
the same seed, split, batch size, learning rate, and 50 epochs as the GRU run.

Development dataset runs use their matching entry points and shape-compatible configs:

```bash
uv run python -m scripts.training.train_image_classification --data mnist --config configs/image_classification/mnist/linear.yaml
uv run python -m scripts.training.train_image_classification --data cifar10 --config configs/image_classification/cifar10/cnn.yaml
```

The best-validation checkpoints are written to `outputs/a1_v1/checkpoints/<run-name>/best.pt`. Training
curves and exact summaries are written to `outputs/a1_v1/results/<run-name>/`.
Resolved configurations are saved in `outputs/a1_v1/configs/`; train/evaluation logs
are saved in `outputs/a1_v1/console/<run-name>/`.

Package optional checkpoint downloads separately for each dataset:

```bash
uv run python -m scripts.release.package_checkpoints --data fashion_mnist
uv run python -m scripts.release.package_checkpoints --data mnist
uv run python -m scripts.release.package_checkpoints --data cifar10
```

Each command writes `outputs/a1_v1/<dataset>-checkpoints.zip`. Upload these archives as GitHub
Release assets; they are not downloaded when cloning the repository. Extract these
archives into `outputs/a1_v1/` to restore their `checkpoints/` subdirectory.

## Evaluation

```bash
uv run python -m scripts.evaluation.evaluate --data fashion_mnist --config configs/image_classification/fashion_mnist/linear.yaml
uv run python -m scripts.evaluation.evaluate --data fashion_mnist --config configs/image_classification/fashion_mnist/mlp.yaml
uv run python -m scripts.evaluation.evaluate --data fashion_mnist --config configs/image_classification/fashion_mnist/lstm.yaml
uv run python -m scripts.evaluation.compare --results-dir outputs/a1_v1/results
```

Each evaluation writes the required confusion matrix, per-class accuracy and support,
most-confused class pairs, and representative correct, confident-error, and uncertain-error
figures under `outputs/a1_v1/results/<run-name>/`.

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
