"""Fixed Matplotlib typography and palette for project figures."""

import matplotlib as mpl
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap, to_hex

LIGHT_BLUE = "#B2DDF4"
BLUE = "#2381C5"
RED = "#E53935"
GREEN = "#359653"
YELLOW = "#D99B24"
COLORS = (BLUE, RED, LIGHT_BLUE, "#F6BDBB", "#549ED0")
BLUE_RED = LinearSegmentedColormap.from_list(
    "blue_red", [BLUE, LIGHT_BLUE, "white", "#F8CECC", RED]
)
CLASS_COLORS = tuple(to_hex(BLUE_RED(i / 9)) for i in range(10))


def apply() -> None:
    """Apply the project style before creating a figure, without requiring LaTeX."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "text.usetex": False,
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "figure.titlesize": 12,
            "figure.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "legend.frameon": False,
            "legend.borderaxespad": 0.3,
            "legend.columnspacing": 1.4,
            "axes.titlepad": 10,
            "axes.labelpad": 5,
            "xtick.direction": "out",
            "ytick.direction": "out",
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
            "lines.linewidth": 1.3,
            "lines.markersize": 4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 400,
            "savefig.bbox": "tight",
        }
    )
