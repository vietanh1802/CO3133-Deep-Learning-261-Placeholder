"""Fixed Matplotlib typography and palette for project figures."""

import matplotlib as mpl
from cycler import cycler

BLUE = "#2878B8"
RED = "#CF4540"
GREEN = "#359653"
YELLOW = "#D99B24"
COLORS = (BLUE, RED, GREEN, YELLOW)


def apply() -> None:
    """Apply the project style before creating a figure, without requiring LaTeX."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "text.usetex": False,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "figure.titlesize": 10,
            "figure.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "axes.axisbelow": True,
            "axes.formatter.use_mathtext": True,
            "axes.prop_cycle": cycler(color=COLORS),
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.minor.width": 0.6,
            "ytick.minor.width": 0.6,
            "lines.linewidth": 1.0,
            "lines.markersize": 4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 400,
            "savefig.bbox": "tight",
        }
    )
