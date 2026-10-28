# CO3133 - Deep Learning and Its Applications (Semester 261)

Course project by Placeholder.

[Website](https://vietanh1802.github.io/CO3133-Deep-Learning-261-Placeholder/) · [Assignment 1](web/assignment1.html) · [Assignment 2](web/assignment2.html) · [Assignment 3](web/assignment3.html)

## Setup

```bash
uv sync --dev
uv run pre-commit install
```

## Datasets

Fashion-MNIST, MNIST, and CIFAR-10 download automatically. Assignment 1 uses Fashion-MNIST.

For SUIM, extract the dataset into:

```text
data/suim/
  train_val/
    images/
    masks/
  TEST/
    images/
    masks/
```

Generate dataset summaries and figures:

```bash
uv run python -m scripts.eda.run_fashion_mnist
uv run python -m scripts.eda.run_mnist
uv run python -m scripts.eda.run_suim
```

EDA outputs go to `results/a1/eda/<dataset>/` or `results/a2/eda/suim/`.
SUIM training reads the committed splits in `results/a2/splits/suim/`.

## Classification

```bash
uv run python -m scripts.training.train_image_classification --data fashion_mnist --config configs/image_classification/fashion_mnist/linear.yaml
uv run python -m scripts.evaluation.evaluate --data fashion_mnist --config configs/image_classification/fashion_mnist/linear.yaml
uv run python -m scripts.evaluation.compare --results-dir results/a1 --epochs 50 \
  --figures-dir web/images/comparison/a1
```

Use matching dataset and configuration paths for MNIST or CIFAR-10.
Configurations live in `configs/image_classification/<dataset>/`.

Train Linear, MLP, CNN, GRU, LSTM, and Transformer on all three datasets
using the shared training entry point:

```bash
bash train.sh
```

Use the shared evaluation command above for each model and dataset.
Place each `best.pt` at the run path named by its configuration.
CNN uses `checkpoints/<dataset>_cnn_spatial_seed42/best.pt`; the other models
use `checkpoints/<dataset>_<model>_seed42/best.pt`.
Evaluation loads the selected checkpoint and saves test outputs in `results/a1/<run-name>/`.

After the evaluations finish, rebuild the comparison table:

```bash
uv run --no-sync python -m scripts.evaluation.compare --results-dir results/a1 --epochs 50
```

The comparison includes only `<dataset>_cnn_spatial_*` CNN runs. Archived
`<dataset>_cnn_seed42` results are excluded even when they remain in `results/a1/`.
There is no fallback to the old CNN when spatial results are unavailable.

## Segmentation

Train the U-Net baseline or resume its latest checkpoint:

```bash
uv run python -m scripts.training.train_segmentation --config configs/segmentation/suim/baseline.yaml
uv run python -m scripts.training.train_segmentation --config configs/segmentation/suim/baseline.yaml --resume
```

For two GPUs:

```bash
uv run torchrun --standalone --nproc-per-node=2 -m scripts.training.train_segmentation --config configs/segmentation/suim/baseline.yaml
```

`batch_size` and `num_workers` are per process. Effective batch size is
`batch_size × accumulation_steps × process_count`. Resume with the same process count.
Use separate run names for separate experiments.

Fine-tune the pretrained DeepLabV3-ResNet50 on the same SUIM splits:

```bash
uv run python -m scripts.training.train_segmentation --config configs/segmentation/suim/pretrained.yaml
```

The first run downloads torchvision's `COCO_WITH_VOC_LABELS_V1` weights. The final
classifier layer is replaced for eight classes and the auxiliary classifier is removed.
AdamW uses learning rates of `1e-4` for the backbone and `1e-3` for the classifier,
with the same warm-up/cosine schedule. BatchNorm running statistics stay fixed to
support small batches; its affine parameters and the rest of the network still train.
The configuration uses batch size 4 and accumulation 4 (effective batch 16 on one GPU).
For CPU checks without weight downloads, construct the model with `pretrained=False`.

After training, export validation metrics and qualitative examples:

```bash
uv run python -m scripts.evaluation.eval_segmentation --evidence results/a2/suim_deeplabv3_ce_seed42/val_evidence.pt --output results/a2/suim_deeplabv3_ce_seed42/evaluation
```

## Outputs

| Path | Contents |
| --- | --- |
| `checkpoints/<run-name>/best.pt` | Selected model |
| `checkpoints/<run-name>/last.pt` | Segmentation resume state |
| `results/a1/<run-name>/` | Classification logs, metrics, and figures |
| `results/a2/<run-name>/` | Segmentation logs, history, summary, curves, and validation evidence |

Classification selects the lowest validation loss; segmentation selects the highest validation mIoU.
Checkpoints are not committed. Dependencies are locked in `uv.lock`; experiment settings and seeds live in the configurations.

Package classification checkpoints for release:

```bash
uv run python -m scripts.release.package_checkpoints --data fashion_mnist
```

This writes `artifacts/fashion_mnist-checkpoints.zip`. MNIST and CIFAR-10 use the same command with their dataset names.

## Source layout

| Package | Purpose |
| --- | --- |
| `src/config` | Configuration parsing and validation |
| `src/data` | Dataset contracts, transforms, and loaders |
| `src/eda` | Statistics, quality checks, duplicates, and splits |
| `src/models` | Model architectures |
| `src/training` | Training loops and checkpoints |
| `src/evaluation` | Metrics and inference costs |
| `src/error` | Prediction error analysis |
| `src/plots` | Figure formatting and rendering |
| `src/utils` | Logging, seeds, environment, and process helpers |
| `scripts` | Command-line entry points |

Shared contracts live in `base.py`; concrete modules name their dataset, task, or model.
Data loading stays independent of EDA, plotting, training, and evaluation.

## Plots

Call `src.plots.format.apply()` before creating a figure. Save with
`src.plots.common.save_figure(figure, output_path)` for 400-dpi PNG and PDF exports.
The shared style uses DejaVu Sans and requires no LaTeX installation.

## Checks

```bash
uv run --locked python -m scripts.check
```

Runs lint, formatting, types, and the CPU test suite. GitHub Actions runs the same
command on pushes and pull requests. Tests use synthetic data without dataset downloads.

- `tests/contracts`: configuration, imports, data, and package boundaries.
- `tests/components`: models, metrics, analysis, plots, and shared training behavior.
- `tests/pipelines`: training, checkpoint reload, resume, evaluation, export, and release packaging.

## Documents

- [Assignment 1 draft](reports/assignment1-draft.md)
- [Assignment 2 proposal](reports/a2_proposal.md)
- [Assignment 2 draft](reports/assignment2-draft.md)
- [AI usage disclosure](AI_USAGE.md)
