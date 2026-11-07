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
from src.plots.format import (
    CLASS_COLORS,
    COLORS,
    CYCLIC,
    DIVERGING,
    QUALITATIVE,
    SEQUENTIAL,
    STYLE,
    apply,
    style_axes,
    text_color,
)


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


def test_apply_restores_reference_theme_without_latex():
    with mpl.rc_context():
        apply()
        expected = dict(mpl.rcParams)
        mpl.rcParams.update({"font.size": 24, "axes.grid": True, "text.usetex": True})
        apply()
        assert dict(mpl.rcParams) == expected
        assert mpl.rcParams["font.family"] == ["monospace"]
        assert mpl.rcParams["font.monospace"] == ["Courier New", "DejaVu Sans Mono"]
        assert mpl.rcParams["mathtext.fontset"] == "dejavusans"
        assert not mpl.rcParams["text.usetex"]
        font = font_manager.findfont(
            font_manager.FontProperties(family="monospace"), fallback_to_default=False
        )
        expected_font = font_manager.FontProperties(fname=font).get_name()
        figure, axis = plt.subplots()
        try:
            axis.set_xlabel(r"Target $R^2$")
            assert axis.xaxis.label.get_fontproperties().get_name() == expected_font
            assert axis.spines["top"].get_visible()
            assert axis.spines["right"].get_visible()
            assert axis.spines["left"].get_linewidth() == 0.6
            style_axes(axis)
            assert any(line.get_visible() for line in axis.get_xgridlines())
            assert axis.get_xgridlines()[0].get_alpha() == 0.25
            assert axis.xaxis.get_minorticklines()
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
        assert image.size == (800, 600)


def test_plotters_apply_style_and_preserve_values(tmp_path, monkeypatch):
    from src.plots import classification, common

    figures = []
    monkeypatch.setattr(classification, "save_figure", lambda figure, path: figures.append(figure))
    monkeypatch.setattr(common, "save_figure", lambda figure, path: figures.append(figure))
    with mpl.rc_context():
        mpl.rcParams["axes.grid"] = True
        plot_learning_curves([1.0, 0.6, 0.4], [1.1, 0.5, 0.7], tmp_path / "curves.png")
        axes = figures[0].axes[0]
        assert axes.lines[0].get_color() == COLORS[0]
        assert axes.lines[1].get_color() == COLORS[1]
        assert list(axes.lines[0].get_ydata()) == [1.0, 0.6, 0.4]
        assert list(axes.lines[2].get_xdata()) == [2, 2]
        assert any(line.get_visible() for line in axes.get_xgridlines())
        plot_confusion_matrix(np.array([[3, 1], [0, 4]]), ["A", "B"], tmp_path / "counts.png")
        image = figures[1].axes[0].images[0]
        assert image.get_cmap().name == "plasma"
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


def test_clustered_similarity_keeps_labels_and_values_aligned(tmp_path, monkeypatch):
    from src.eda.classification import profile_dataset
    from src.plots import eda

    images = np.arange(4 * 3 * 3, dtype=np.uint8).reshape(4, 3, 3)
    profile = profile_dataset(
        images, np.arange(4), ["A", "B", "C", "D"], dataset="toy", split="train"
    )
    figures = []
    monkeypatch.setattr(eda, "save_figure", lambda figure, path: figures.append(figure))
    eda.plot_class_similarity(profile, tmp_path / "similarity.png", title="Similarity")
    tree, matrix, _ = figures[0].axes
    assert matrix.images[0].get_cmap().name == "PuOr_r"
    labels = [tick.get_text() for tick in matrix.get_yticklabels()]
    assert labels == [tick.get_text() for tick in matrix.get_xticklabels()]
    order = [list(profile.balance.class_names).index(label) for label in labels]
    np.testing.assert_allclose(
        matrix.images[0].get_array(), profile.class_similarity[np.ix_(order, order)]
    )
    assert tree.yaxis_inverted()
    assert tree.get_ylim() == (40, 0)
    assert matrix.get_ylim() == (3.5, -0.5)


def test_palette_roles_are_distinct_and_annotation_contrast_matches_fill():
    assert SEQUENTIAL.name == "plasma"
    assert DIVERGING.name == "PuOr_r"
    assert CYCLIC.name == "twilight"
    assert QUALITATIVE.name == "tab10"
    assert len(set(CLASS_COLORS)) == 10
    assert tuple(STYLE["palette"]["methods"]) == COLORS
    assert text_color(SEQUENTIAL(0.0)) == "white"
    assert text_color(SEQUENTIAL(1.0)) == "black"
    assert text_color(DIVERGING(0.5)) == "black"
    with mpl.rc_context():
        apply("heatmap")
        assert not mpl.rcParams["axes.grid"]
        apply()
        assert mpl.rcParams["axes.grid"]
    with pytest.raises(ValueError, match="line or heatmap"):
        apply("unknown")
