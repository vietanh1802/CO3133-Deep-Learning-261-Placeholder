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

The best-validation checkpoints are stored together in `checkpoints/a1.zip`, with
`checkpoints/<run-name>/best.pt` inside the ZIP. Each completed training run updates
only its own checkpoint in the archive. Evaluation reads that member directly; no
manual extraction is needed. Training logs, curves, and summaries are written to
`results/a1/<run-name>/`. ZIP packaging occurs after the timed training loop.

The repository keeps results and checkpoint archives separate:

```text
results/a1/<run-name>/
results/a1/eda/
results/a1/archive/            # Earlier draft runs, excluded from comparisons
checkpoints/a1.zip
```

A2 and A3 can use `results/a2/`, `results/a3/`, `checkpoints/a2.zip`,
and `checkpoints/a3.zip`. This change only implements the A1 pipeline.
Checkpoint ZIPs, logs, and evidence tensors remain ignored by Git.

To package an existing folder of downloaded `<run-name>/best.pt` files:

```bash
uv run python -m scripts.release.package_checkpoints --checkpoint-dir .tmp/drive-a1/checkpoints --output-dir checkpoints
```

This creates `checkpoints/a1.zip`. Running the command without arguments can
repackage the existing local archive. Optional `--data fashion_mnist` produces
a dataset-only ZIP for sharing.

## Copy existing Colab results back to local

The running Colab notebook can finish using its current Drive path:
`MyDrive/BTL_DL_Result/VanhResult/a1_v1/`. No retraining is required for this
layout change. Wait until all six models finish training and testing, then run the
comparison/ZIP cell so that `a1_cuda_outputs.zip` contains the final outputs.

1. Download and extract `a1_cuda_outputs.zip` into `.tmp/drive-a1/`. Locate the
   extracted folder that contains `checkpoints/` and `results/`.
2. Copy the **contents** of that extracted `results/` folder into local
   `results/a1/`. Replace matching files in the six Fashion-MNIST run folders;
   copy the new `comparison_test.md` and `comparison_test.csv` too. Do not create
   `results/a1/results/`. Existing EDA, MNIST, and CIFAR-10 results can remain.
3. In File Explorer, compress only the extracted `checkpoints/` folder into a
   ZIP, rename it `a1.zip`, and paste it into local `checkpoints/`, replacing
   the previous `a1.zip`. Both `checkpoints/<run-name>/best.pt` and
   `<run-name>/best.pt` member layouts are supported. Alternatively, use the
   packaging command above. Do not rename the full `a1_cuda_outputs.zip` as a
   checkpoint-only archive.
4. Optionally copy each extracted `console/<run-name>/train.log` and
   `evaluate.log` into `results/a1/<run-name>/`. The Colab `configs/` folder
   records its own absolute runtime paths; do not overwrite the source YAML files
   under the repository's `configs/image_classification/` with those copies.
5. Run `uv run python -m scripts.evaluation.compare --results-dir results/a1`
   to regenerate the comparison table. If testing already finished on Colab, do
   not rerun local evaluation just to import the results; that would replace the
   recorded inference timings with timings from the local device.

Saved JSON/checkpoint metadata retains its original runtime paths as provenance.
Current commands use the repository YAML paths and the run name to locate the
imported artifacts. Files from earlier experiments should be backed up before
replacement; the local files moved during this layout correction are preserved
under `.tmp/before-separate-a1-folders/`.

## Evaluation

```bash
uv run python -m scripts.evaluation.evaluate --data fashion_mnist --config configs/image_classification/fashion_mnist/linear.yaml
uv run python -m scripts.evaluation.evaluate --data fashion_mnist --config configs/image_classification/fashion_mnist/mlp.yaml
uv run python -m scripts.evaluation.evaluate --data fashion_mnist --config configs/image_classification/fashion_mnist/lstm.yaml
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
