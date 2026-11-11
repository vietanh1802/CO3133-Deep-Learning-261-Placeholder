"""Plot per-class RGB and brightness distributions from saved SUIM validation images.

Run: uv run python -m scripts.eda.plot_suim_colour
Uses resized, unnormalized RGB images and ground-truth labels, not mask colours.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

from src.data.suim import CLASS_NAMES  # noqa: E402
from src.plots.common import save_figure  # noqa: E402
from src.plots.format import apply  # noqa: E402

EVIDENCE = Path("results/a2/suim_ablation_reference_seed42/val_evidence.pt")
OUTPUT = Path("results/a2/eda/suim")


def main() -> None:
    """Pool pixels by ground-truth class and normalize each distribution separately."""
    evidence = torch.load(EVIDENCE, weights_only=True, map_location="cpu")
    images = evidence["images"].numpy()
    targets = evidence["targets"].numpy()
    assert evidence["split"] == "validation"
    assert images.shape == (287, 3, 240, 320) and targets.shape == (287, 240, 320)
    assert np.isfinite(images).all() and 0 <= images.min() <= images.max() <= 1
    assert targets.min() >= 0 and targets.max() < 8
    rgb = np.moveaxis(images, 1, -1)
    edges = np.linspace(0, 1, 33)
    centers = (edges[:-1] + edges[1:]) / 2
    apply()
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )
    fig, axes = plt.subplots(2, 4, figsize=(9, 4.3), sharex=True, sharey=True)
    names = ("Water", "Divers", "Plants", "Wrecks", "Robots", "Reefs", "Fish", "Sea floor")
    records = {}
    total_pixels = 0
    for k, ax in enumerate(axes.flat):
        pixels = rgb[targets == k]
        assert len(pixels) > 0
        total_pixels += len(pixels)
        # Gamma-encoded RGB luma is a brightness proxy, not physical luminance.
        brightness = pixels @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        distributions = [pixels[:, i] for i in range(3)] + [brightness]
        record = {"pixels": len(pixels), "histograms": {}}
        for values, label, color in zip(
            distributions,
            ("Red", "Green", "Blue", "Brightness"),
            ("#cf5b5b", "#52946b", "#5187bd", "#3d3d3d"),
            strict=True,
        ):
            counts, _ = np.histogram(values, bins=edges)
            assert counts.sum() == len(pixels)
            frequency = counts / len(pixels)
            assert np.isclose(frequency.sum(), 1)
            ax.plot(
                centers,
                frequency,
                color=color,
                linewidth=1.4,
                linestyle="--" if label == "Brightness" else "-",
                label=label,
            )
            record["histograms"][label] = frequency.tolist()
        records[CLASS_NAMES[k]] = record
        ax.set_title(f"{CLASS_NAMES[k]} · {names[k]}", pad=6)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1])
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.grid(visible=False, which="minor")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.53, 1.015)
    )
    fig.supxlabel("Image intensity (0 = dark, 1 = bright)", y=0.055, fontsize=11)
    fig.supylabel("Share of class pixels per bin", x=0.012, fontsize=11)
    fig.text(
        0.53,
        0.005,
        "287 validation images · 240 × 320 · normalized within each class",
        ha="center",
        fontsize=9,
        color="0.35",
    )
    fig.subplots_adjust(left=0.09, right=0.97, top=0.86, bottom=0.19, hspace=0.43, wspace=0.18)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    save_figure(fig, OUTPUT / "class_colour_distribution.png")
    plt.close(fig)
    assert total_pixels == targets.size
    (OUTPUT / "class_colour_distribution.json").write_text(
        json.dumps(
            {
                "source": str(EVIDENCE),
                "split": "validation",
                "images": len(images),
                "resolution_hw": [240, 320],
                "bin_edges": edges.tolist(),
                "weighting": "pooled pixels; each class and channel normalized separately",
                "brightness": "0.2126 R + 0.7152 G + 0.0722 B on gamma-encoded RGB",
                "classes": records,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
