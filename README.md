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

The analysis itself lives in `src/eda/classification.py` (statistics), `src/plots/eda.py`
(figures), and `src/eda/report.py` (one shared run), so the entry points only have
to name a dataset.

### Assignment 2: SUIM

SUIM is not downloaded automatically and is not in the repository. Unpack it so that `data/suim/train_val/` and `data/suim/TEST/` exist, each with an `images/` and a `masks/` folder, then run:

```bash
uv run python -m scripts.eda.run_suim
```

One run produces every number quoted in the A2 dataset proposal. It takes a few minutes, most of it the near-duplicate search over all 1,635 pairs.

| Output                                             | What it answers                                                                                                                                                 |
| -------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `results/a2/eda/suim/summary.json`                 | Pairs on disk, decode and size faults, per-class pixel and image counts for both splits, the resolution histogram, and every duplicate group listed by filename |
| `results/a2/eda/suim/class_distribution.png`       | Image count beside pixel share, which rank the classes differently                                                                                              |
| `results/a2/eda/suim/region_area_distribution.png` | How much of a frame a class covers when present, separating rare-and-large from common-and-small                                                                |
| `results/a2/eda/suim/resolution_distribution.png`  | The eleven resolutions on a log scale                                                                                                                           |
| `results/a2/splits/suim/train.txt`, `val.txt`      | The fixed 1,143 / 287 split, one filename stem per line                                                                                                         |

The two split files are committed on purpose. Training reads them rather than re-deriving the split, so a later change to the code cannot silently move an image between sides. The statistics live in `src/data/suim.py`; the script only orchestrates and writes.

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

## Assignment 2 — segmentation

SUIM semantic segmentation proposal is approved. Train the U-Net baseline (trained from
scratch) on the committed split with:

```bash
uv run python -m scripts.training.train_segmentation --config configs/segmentation/suim/baseline.yaml
uv run python -m scripts.training.train_segmentation --config configs/segmentation/suim/baseline.yaml --resume
```

The second command continues an interrupted run from `checkpoints/<name>/last.pt`. The
checkpoint with the highest validation mIoU is kept as `checkpoints/<name>/best.pt`.
Checkpoints are not committed; rerun the command to rebuild them. Every run writes
`results/a2/<name>/`:

| Output                   | Contents                                                                                                                                                                     |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `train.log`              | Configuration, device, and per-epoch loss, mIoU and mDice                                                                                                                    |
| `history.json`           | Per-epoch records, rewritten after every epoch                                                                                                                               |
| `training_summary.json`  | Validation metrics of the selected checkpoint (all, boundary, interior), parameter count, hardware, software versions, git commit, seed, training time, peak training memory, inference cost |
| `learning_curves.png`    | Loss and validation mIoU per epoch, with the selected epoch marked                                                                                                           |
| `val_evidence.pt`        | Validation predictions for `scripts.evaluation.eval_segmentation` (not committed)                                                                                            |

`training_seconds` includes each epoch's validation pass. Inference cost times only the
forward pass of the selected checkpoint over the validation split, after two warm-up
batches.

- Data loading: `src/data/suim.py`
- Models: `src/models/segmentation.py` (U-Net; pretrained DeepLabV3 in progress)
- Training: `src/training/segmentation.py`; metrics: `src/evaluation/segmentation.py`;
  inference cost: `src/evaluation/compute_cost.py`
- Configurations: `configs/segmentation/suim/`
- Draft: [Assignment 2 M2 outline](reports/assignment2-draft.md)

## Train A2 on multiple GPUs

Launch one process per GPU on a single machine with the existing configuration:

```bash
uv run torchrun --standalone --nproc-per-node=2 -m scripts.training.train_segmentation --config configs/segmentation/suim/baseline.yaml
```

If automatic rendezvous fails to resolve a local address, use a static IPv4
coordinator. Choose a free port:

```bash
uv run torchrun --rdzv-backend=static --master-addr=127.0.0.1 --master-port=29571 --nproc-per-node=2 -m scripts.training.train_segmentation --config configs/segmentation/suim/baseline.yaml
```

Add `--resume` to continue from `last.pt`. Resume requires the same process count.
CPU, MPS, and ordinary single-GPU commands remain available without `torchrun`.
A CPU-only `torchrun` uses Gloo for distributed smoke tests.

`batch_size` and `num_workers` apply to each process. For full accumulation groups,
the effective batch is `batch_size * accumulation_steps * process_count`.
The baseline settings therefore use 32 images per update on two GPUs.
Set `accumulation_steps: 2` in a separate configuration to retain an effective batch
of 16. The learning rate is not scaled automatically. Each model keeps local
BatchNorm statistics; DDP does not make the run identical to one larger GPU batch.

Each process trains a shard of the existing training split. The sampler may repeat
up to `process_count - 1` images to give processes equal work. The single progress
bar counts processed images across all processes, including those repetitions.
It updates at optimizer boundaries and shows the global sample-weighted loss.
Gradients accumulate locally and synchronize on the last backward pass of each
accumulation group, including the final partial group.

Rank 0 validates the complete validation split and shares its scores with the
other processes. It writes `train.log`, history, checkpoints, plots, evidence,
and the final summary. Other ranks write warnings and failures to `rank<N>.log`.
Worker failures also propagate through `torchrun`.
Checkpoint model weights load without DDP for evaluation. Distributed checkpoints
also store each rank's random state for resume. The summary records process count
and effective batch size; reported inference and peak memory describe rank 0.
Use a different run name for separate experiments so their artifacts do not overlap.

## Shared training code

A1 classification and A2 segmentation use `src/training/epoch.py` for optimizer
updates, gradient accumulation, scheduler steps, and distributed progress.
Classification supplies a correct-prediction counter. Segmentation keeps pixel
validation and best-mIoU selection in `src/training/segmentation.py`.
The classification entry point still uses its original single-process setup.

`src/training/checkpoint.py` owns atomic checkpoint writes and state restoration.
Task-specific wrappers preserve their metadata and load older single-process
checkpoints. `src/config/io.py` shares YAML validation and path conversion
while each task retains its own configuration fields and defaults.

## Source organization

Packages own one responsibility. A `base.py` declares a real contract or registry.
Shared implementations have descriptive names. Concrete modules name their task,
dataset, or model; adding one does not require copying a training loop.

| Package | Responsibility | Shared code | Concrete code |
| --- | --- | --- | --- |
| `data` | Read samples, transform tensors, and build loaders | `base.py`, `transform.py`, `classification.py`, `segmentation.py` | `mnist.py`, `fashion_mnist.py`, `cifar10.py`, `suim.py` |
| `eda` | Inspect datasets, check quality and duplicates, and prepare splits | `base.py`, `report.py` | `classification.py`, `suim.py` |
| `models` | Define model components and architectures | `base.py` | `cnn.py`, `gru.py`, `unet.py`, and other models |
| `training` | Update weights, validate, and save training state | `epoch.py`, `checkpoint.py` | `classification.py`, `segmentation.py` |
| `evaluation` | Compute task scores and inference costs | `compute_cost.py` | `classification.py`, `classification_metrics.py`, `segmentation.py` |
| `plots` | Render figures from results | `common.py` | `classification.py`, `segmentation.py`, `eda.py` |
| `config` | Validate experiment settings | `io.py` | `classification.py`, `segmentation.py` |
| `error` | Analyze prediction failures | `base.py`, `selection.py` | `classification.py` |
| `utils` | Provide task-independent runtime helpers | `distributed.py`, `logging.py`, `seed.py`, `environment.py` | None |

Data loading does not import EDA, plots, training, or evaluation.
EDA uses the data loaders and decoding utilities. Training and evaluation use
models and data; their scripts compose plotting and artifact export.
Package initializers do not present one concrete task as a general-purpose API.
Create another contract only when callers need a shared interface.

`data/segmentation.py` owns the reusable segmentation dataset and data loaders.
`SegmentationDataset` shares image loading, resizing, paired augmentation,
normalization, and evidence display support. Concrete datasets implement
`decode_target` for their mask encoding. `data/suim.py` supplies SUIM file pairing,
committed split selection, RGB mask decoding, and class definitions, then delegates
to the shared dataset and loader implementations.

## Plot formatting

All figures use `src/plots/format.py`. Call `apply()` before creating a figure to
apply the fixed DejaVu Sans text and matching math fonts, restrained palette, thin axes,
and grid-free styling. Importing the format module does not apply the style.
No external LaTeX installation is needed.

Save through `src.plots.common.save_figure(figure, output_path)`. Existing PNG
filenames are retained, with 400-dpi PNG and matching PDF exports. PDFs embed
TrueType fonts. Dataset mask colors and scientific heatmap scales are preserved.

Generate isolated synthetic previews without training or replacing experiment
figures:

```bash
uv run python -m scripts.plots.preview_style --output-dir /tmp/co3133-plot-preview
```
