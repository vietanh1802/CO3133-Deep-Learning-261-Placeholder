"""Check one-factor designs, trainable parameters, differentiable loss and common grids."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn
from torchvision.models.segmentation import DeepLabV3_ResNet101_Weights

from scripts.evaluation.ablate_segmentation import ablation_configs
from src.config.segmentation import load_segmentation_config
from src.evaluation.segmentation import evaluate_model
from src.models import DeepLabV3
from src.training.losses import build_segmentation_loss
from src.training.segmentation import build_segmentation_optimizer

ROOT = Path(__file__).resolve().parents[2]


def test_design_changes_only_one_factor_and_preserves_reference():
    base = load_segmentation_config(ROOT / "configs/segmentation/suim/ablation.yaml")
    before = base.to_dict()
    configs = ablation_configs(base, [42, 43])
    assert len({config.name for config in configs}) == 12
    for start in (0, 6):
        reference = configs[start].to_dict()
        for config, expected in zip(
            configs[start + 1 : start + 6],
            (
                {"model_config"},
                {"augment"},
                {"model_config"},
                {"loss"},
                {"image_height", "image_width"},
            ),
            strict=True,
        ):
            changed = {
                key
                for key, value in config.to_dict().items()
                if key != "name" and value != reference[key]
            }
            assert changed == expected
            assert (config.evaluation_height, config.evaluation_width) == (240, 320)
        assert configs[start + 1].model_config == {
            **reference["model_config"],
            "freeze_backbone": True,
        }
        assert configs[start + 3].model_config == {
            **reference["model_config"],
            "backbone": "resnet101",
        }
    assert base.to_dict() == before


@pytest.mark.parametrize("seeds", [[], [42, 42], [-1], [2**32]])
def test_invalid_seeds_are_rejected(seeds):
    base = load_segmentation_config(ROOT / "configs/segmentation/suim/ablation.yaml")
    with pytest.raises(ValueError, match="seeds"):
        ablation_configs(base, seeds)


@pytest.mark.parametrize(
    "changes",
    [
        {"loss": "unknown"},
        {"augment": "false"},
        {"evaluation_height": None},
        {"evaluation_height": 0, "evaluation_width": 12},
    ],
)
def test_invalid_ablation_settings_are_rejected(changes):
    base = load_segmentation_config(ROOT / "configs/segmentation/suim/ablation.yaml")
    with pytest.raises(ValueError):
        replace(base, **changes)


def test_resnet101_selects_matching_pretraining(tiny_deeplab):
    DeepLabV3(backbone="resnet101")
    tiny_deeplab.assert_called_once_with(
        weights=DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1, weights_backbone=None
    )
    with pytest.raises(ValueError, match="backbone must"):
        DeepLabV3(backbone="invalid")


def test_frozen_backbone_stays_unchanged_while_head_updates(tiny_deeplab):
    model = DeepLabV3(pretrained=False, freeze_backbone=True)
    before = {key: value.clone() for key, value in model.network.backbone.state_dict().items()}
    head_before = model.network.classifier[-1].weight.detach().clone()
    optimizer = build_segmentation_optimizer(
        model, learning_rate=0.01, backbone_learning_rate=0.001, weight_decay=0.01
    )
    model.eval().train()
    logits = model(torch.randn(2, 3, 8, 12))
    build_segmentation_loss("cross_entropy")(
        logits, torch.zeros(2, 8, 12, dtype=torch.long)
    ).backward()
    optimizer.step()
    assert not model.network.backbone.training
    assert all(parameter.grad is None for parameter in model.network.backbone.parameters())
    for key, value in model.network.backbone.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    assert not torch.equal(head_before, model.network.classifier[-1].weight)


def test_combined_loss_rewards_correct_labels_and_has_finite_gradients():
    target = torch.tensor([[[0, 1], [1, 0]]])
    logits = torch.randn(1, 8, 2, 2, requires_grad=True)
    loss = build_segmentation_loss("ce_dice")(logits, target)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0
    correct = torch.full((1, 8, 2, 2), -20.0).scatter_(1, target.unsqueeze(1), 20.0)
    incorrect = correct.roll(1, dims=1)
    assert build_segmentation_loss("ce_dice")(correct, target) < 1e-5
    assert build_segmentation_loss("ce_dice")(incorrect, target) > 1
    assert isinstance(build_segmentation_loss("cross_entropy"), nn.CrossEntropyLoss)
    with pytest.raises(ValueError, match="loss must"):
        build_segmentation_loss("other")


def test_low_resolution_scores_are_resized_before_argmax_on_common_grid():
    class Scores(nn.Module):
        def forward(self, images):
            return images

    logits = torch.tensor([[[[4.0, 0.0]], [[0.0, 4.0]]]])
    target = torch.tensor([[[0, 0, 1, 1], [0, 0, 1, 1]]])
    result = evaluate_model(Scores(), [(logits, target)], torch.device("cpu"), num_classes=2)
    assert result["all"].confusion_matrix.tolist() == [[4, 0], [0, 4]]
    assert result["all"].mean_iou == 1.0
