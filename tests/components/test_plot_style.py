"""Publication style is explicit, repeatable, and shared by actual figures."""

import subprocess
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib import font_manager
from PIL import Image

from src.plots.classification import plot_learning_curves
from src.plots.common import plot_confusion_matrix, save_figure
from src.plots.format import BLUE, RED, apply


def test_importing_format_does_not_change_style():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import matplotlib as m; m.rcParams['font.size']=17; "
            "m.rcParams['font.family']=['monospace']; "
            "before={k:str(v) for k,v in m.rcParams.items() if k != 'backend'}; "
            "import src.plots.format; "
            "assert before=={k:str(v) for k,v in m.rcParams.items() if k != 'backend'}",
        ],
        check=True,
        timeout=30,
    )


def test_apply_restores_fixed_sans_serif_settings_without_latex():
    with mpl.rc_context():
        apply()
        expected = dict(mpl.rcParams)
        mpl.rcParams.update({"font.size": 24, "axes.grid": True, "text.usetex": True})
        apply()
        assert dict(mpl.rcParams) == expected
        assert mpl.rcParams["font.family"] == ["sans-serif"]
        assert mpl.rcParams["font.sans-serif"][0] == "DejaVu Sans"
        assert mpl.rcParams["mathtext.fontset"] == "dejavusans"
        assert not mpl.rcParams["text.usetex"]
        font = font_manager.findfont(
            font_manager.FontProperties(family="DejaVu Sans"), fallback_to_default=False
        )
        assert "DejaVuSans" in font
        figure, axis = plt.subplots()
        try:
            axis.set_xlabel(r"Target $R^2$")
            assert axis.xaxis.label.get_fontproperties().get_name() == "DejaVu Sans"
            assert not axis.spines["top"].get_visible()
            assert not axis.spines["right"].get_visible()
            assert axis.spines["left"].get_linewidth() == 0.6
            assert not any(line.get_visible() for line in axis.get_xgridlines())
            figure.canvas.draw()
        finally:
            plt.close(figure)


@pytest.mark.parametrize("suffix", [".png", ".pdf", ""])
def test_save_figure_writes_pdf_and_400dpi_png(tmp_path, suffix):
    with mpl.rc_context():
        apply()
        figure, axis = plt.subplots(figsize=(2, 1.5))
        try:
            axis.plot([0, 1], [0, 1])
            save_figure(figure, tmp_path / f"figure{suffix}")
        finally:
            plt.close(figure)
    assert (tmp_path / "figure.pdf").read_bytes().startswith(b"%PDF-")
    with Image.open(tmp_path / "figure.png") as image:
        assert image.info["dpi"] == pytest.approx((400, 400), abs=0.1)


def test_plotters_apply_style_and_preserve_values(tmp_path, monkeypatch):
    from src.plots import classification, common

    figures = []
    monkeypatch.setattr(classification, "save_figure", lambda figure, path: figures.append(figure))
    monkeypatch.setattr(common, "save_figure", lambda figure, path: figures.append(figure))
    with mpl.rc_context():
        mpl.rcParams["axes.grid"] = True
        plot_learning_curves([1.0, 0.6, 0.4], [1.1, 0.5, 0.7], tmp_path / "curves.png")
        axes = figures[0].axes[0]
        assert axes.lines[0].get_color() == BLUE
        assert axes.lines[1].get_color() == RED
        assert list(axes.lines[0].get_ydata()) == [1.0, 0.6, 0.4]
        assert list(axes.lines[2].get_xdata()) == [2, 2]
        assert not any(line.get_visible() for line in axes.get_xgridlines())
        plot_confusion_matrix(np.array([[3, 1], [0, 4]]), ["A", "B"], tmp_path / "counts.png")
        image = figures[1].axes[0].images[0]
        assert image.get_cmap().name == "Blues"
        assert image.get_clim() == (0.0, 1.0)
        np.testing.assert_allclose(image.get_array(), [[0.75, 0.25], [0, 1]])


def test_shared_predictions_pair_images_with_captions_in_two_columns(tmp_path, monkeypatch):
    from src.plots import classification

    figures = []
    monkeypatch.setattr(classification, "save_figure", lambda figure, path: figures.append(figure))
    samples = [
        {"sample_index": i, "target": 0, "category": "Both miss", "predictions": {"CNN": 1}}
        for i in range(3)
    ]
    classification.plot_shared_predictions(
        [np.zeros((28, 28)) for _ in samples], samples, ["A", "B"], tmp_path / "shared.png"
    )
    figure = figures[0]
    figure.canvas.draw()
    assert figure.get_figheight() == pytest.approx(3.4)
    images, captions = figure.axes[::2], figure.axes[1::2]
    assert images[0].get_position().x0 < images[1].get_position().x0
    assert images[2].get_position().y1 < images[0].get_position().y0
    for index, (image, caption) in enumerate(zip(images, captions, strict=True)):
        assert image.get_position().x1 < caption.get_position().x0
        assert image.get_title() == f"#{index}: A"
        assert caption.texts[0].get_text() == "Both miss\nCNN: B (wrong)"
