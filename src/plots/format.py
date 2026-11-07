"""Apply the project theme and semantic palettes from configs/plotting.toml."""

import tomllib
from pathlib import Path

import matplotlib as mpl
from cycler import cycler
from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.colors import to_hex
from matplotlib.ticker import AutoMinorLocator

with (Path(__file__).resolve().parents[2] / "configs/plotting.toml").open("rb") as stream:
    STYLE = tomllib.load(stream)

COLORS = tuple(STYLE["palette"]["methods"])
SEQUENTIAL = colormaps[STYLE["colormaps"]["sequential"]]
DIVERGING = colormaps[STYLE["colormaps"]["diverging"]]
CYCLIC = colormaps[STYLE["colormaps"]["cyclic"]]
QUALITATIVE = colormaps[STYLE["colormaps"]["qualitative"]]
CLASS_COLORS = tuple(to_hex(QUALITATIVE(i)) for i in range(QUALITATIVE.N))


def apply(kind: str = "line") -> None:
    """Apply typography and defaults before creating a line or heatmap figure."""
    if kind not in ("line", "heatmap"):
        raise ValueError("kind must be line or heatmap")
    font, axes, figure = STYLE["typography"], STYLE["axes"], STYLE["figure"]
    major_grid = STYLE["grid"]["major"]
    mpl.rcParams.update(
        {
            "font.family": "monospace",
            "font.monospace": font["family"],
            "mathtext.fontset": "dejavusans",
            "text.usetex": False,
            "font.size": font["base_size"],
            "axes.titlesize": font["title_size"],
            "axes.labelsize": font["axis_label_size"],
            "figure.titlesize": font["title_size"],
            "figure.labelsize": font["axis_label_size"],
            "xtick.labelsize": font["tick_label_size"],
            "ytick.labelsize": font["tick_label_size"],
            "legend.fontsize": font["legend_size"],
            "legend.frameon": STYLE["legend"]["frameon"],
            "legend.handlelength": STYLE["legend"]["handlelength"],
            "xtick.direction": axes["tick_direction"],
            "ytick.direction": axes["tick_direction"],
            "figure.figsize": [figure["width"], figure["height"]],
            "figure.facecolor": figure["facecolor"],
            "axes.facecolor": figure["facecolor"],
            "savefig.facecolor": figure["facecolor"],
            "axes.linewidth": axes["linewidth"],
            "axes.spines.top": axes["full_frame"],
            "axes.spines.right": axes["full_frame"],
            "axes.grid": kind == "line" and major_grid["visible"],
            "axes.grid.which": "major",
            "grid.color": major_grid["color"],
            "grid.linewidth": major_grid["linewidth"],
            "grid.alpha": major_grid["alpha"],
            "axes.axisbelow": axes["grid_below_data"],
            "axes.formatter.use_mathtext": True,
            "axes.prop_cycle": cycler(color=COLORS),
            "image.cmap": SEQUENTIAL.name,
            "xtick.major.width": axes["tick_width"],
            "ytick.major.width": axes["tick_width"],
            "xtick.minor.width": axes["tick_width"],
            "ytick.minor.width": axes["tick_width"],
            "lines.linewidth": STYLE["line"]["linewidth"],
            "lines.markersize": STYLE["line"]["markersize"],
            "errorbar.capsize": STYLE["errorbar"]["capsize"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": figure["export_dpi"],
            "savefig.bbox": None if figure["bbox_inches"] == "standard" else "tight",
        }
    )


def style_axes(axis: Axes) -> None:
    """Apply the reference grids to charts, leaving images and colorbars clear."""
    if not axis.axison or axis.images or axis.get_label() == "<colorbar>":
        axis.grid(False, which="both")
        return
    if not mpl.rcParams["axes.grid"]:
        return
    if axis.get_xscale() == "linear":
        axis.xaxis.set_minor_locator(AutoMinorLocator(2))
    if axis.get_yscale() == "linear":
        axis.yaxis.set_minor_locator(AutoMinorLocator(2))
    for which in ("major", "minor"):
        grid = STYLE["grid"][which]
        axis.grid(
            grid["visible"],
            which=which,
            color=grid["color"],
            linewidth=grid["linewidth"],
            alpha=grid["alpha"],
        )


def text_color(color: tuple[float, float, float, float]) -> str:
    """Choose black or white annotations by the fill's relative luminance."""
    linear = [
        value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4
        for value in color[:3]
    ]
    luminance = sum(
        value * weight for value, weight in zip(linear, (0.2126, 0.7152, 0.0722), strict=True)
    )
    return "black" if luminance > 0.179 else "white"
