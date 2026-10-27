"""DeepLabV3 tensor contracts, transfer learning, and optimizer coverage."""

import pytest
import torch
from torch import nn
from torchvision.models import ResNet50_Weights
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights

from src.models import DeepLabV3, UNet, build_model
from src.training.segmentation import build_segmentation_optimizer, warmup_cosine


def test_real_deeplab_trains_on_one_image_and_preserves_resolution(monkeypatch):
    def unexpected_download(*args, **kwargs):
        raise AssertionError("pretrained=False must not download weights")

    monkeypatch.setattr(DeepLabV3_ResNet50_Weights, "get_state_dict", unexpected_download)
    monkeypatch.setattr(ResNet50_Weights, "get_state_dict", unexpected_download)
    model = build_model("deeplabv3", {"num_classes": 8, "pretrained": False})
    assert isinstance(model, DeepLabV3)
    assert model.output_dim == 8
    assert model.network.aux_classifier is None
    norms = [module for module in model.modules() if isinstance(module, nn.BatchNorm2d)]
    statistics = []
    for module in norms:
        assert module.running_mean is not None and module.running_var is not None
        statistics.append((module.running_mean.clone(), module.running_var.clone()))
    model.train()

    images = torch.randn(1, 3, 32, 48)
    targets = torch.randint(0, 8, (1, 32, 48))
    logits = model(images)
    assert logits.shape == (1, 8, 32, 48)
    assert logits.dtype == torch.float32
    assert logits.argmax(1).shape == targets.shape
    loss = nn.CrossEntropyLoss()(logits, targets)
    assert torch.isfinite(loss)
    loss.backward()
    for parameter in (model.network.backbone.conv1.weight, model.network.classifier[-1].weight):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    for module, (mean, variance) in zip(norms, statistics, strict=True):
        assert not module.training
        assert module.weight.requires_grad
        assert module.weight.grad is not None
        torch.testing.assert_close(module.running_mean, mean, rtol=0, atol=0)
        torch.testing.assert_close(module.running_var, variance, rtol=0, atol=0)

    model.eval()
    with torch.no_grad():
        assert model(torch.zeros(1, 3, 240, 320)).shape == (1, 8, 240, 320)


@pytest.mark.parametrize("pretrained", [True, False])
def test_constructor_selects_weights_and_replaces_only_the_output_layer(tiny_deeplab, pretrained):
    model = DeepLabV3(num_classes=8, pretrained=pretrained)
    tiny_deeplab.assert_called_once_with(
        weights=DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1 if pretrained else None,
        weights_backbone=None,
    )
    assert model.network.classifier[-1].out_channels == 8
    assert model.network.aux_classifier is None


@pytest.mark.parametrize("freeze_batch_norm", [True, False])
def test_batch_norm_policy_survives_eval_then_train(tiny_deeplab, freeze_batch_norm):
    model = DeepLabV3(pretrained=False, freeze_batch_norm=freeze_batch_norm)
    assert model.eval() is model
    assert all(not module.training for module in model.modules())
    assert model.train() is model
    assert model.network.training
    norms = [module for module in model.modules() if isinstance(module, nn.BatchNorm2d)]
    assert all(module.training is not freeze_batch_norm for module in norms)
    assert all(parameter.requires_grad for parameter in model.parameters())


@pytest.mark.parametrize("num_classes", [0, -1])
def test_deeplab_rejects_invalid_class_count(num_classes):
    with pytest.raises(ValueError, match="num_classes must be positive"):
        DeepLabV3(num_classes=num_classes)


@pytest.mark.parametrize("shape", [(3, 16, 16), (2, 1, 16, 16), (2, 4, 16, 16)])
def test_deeplab_rejects_non_rgb_batches(tiny_deeplab, shape):
    model = DeepLabV3(pretrained=False)
    with pytest.raises(ValueError, match="images must have shape"):
        model(torch.zeros(shape))


def test_optimizer_covers_every_parameter_once_and_preserves_rate_ratio(tiny_deeplab):
    model = DeepLabV3(pretrained=False)
    optimizer = build_segmentation_optimizer(
        model, learning_rate=1e-3, backbone_learning_rate=1e-4, weight_decay=1e-4
    )
    head, backbone = optimizer.param_groups
    assert [head["name"], backbone["name"]] == ["head", "backbone"]
    assert [head["lr"], backbone["lr"]] == pytest.approx([1e-3, 1e-4])
    head_ids = {id(parameter) for parameter in head["params"]}
    backbone_ids = {id(parameter) for parameter in backbone["params"]}
    assert not head_ids & backbone_ids
    assert head_ids | backbone_ids == {id(parameter) for parameter in model.parameters()}
    assert all(group["weight_decay"] == 1e-4 for group in optimizer.param_groups)
    scheduler = warmup_cosine(optimizer, warmup_steps=2, total_steps=5)
    for _ in range(5):
        assert head["lr"] == pytest.approx(10 * backbone["lr"])
        optimizer.step()
        scheduler.step()


def test_unet_keeps_one_optimizer_group():
    model = UNet(base_width=2)
    optimizer = build_segmentation_optimizer(model, learning_rate=1e-3, weight_decay=1e-4)
    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["lr"] == 1e-3
    with pytest.raises(ValueError, match="only supported for DeepLabV3"):
        build_segmentation_optimizer(
            model, learning_rate=1e-3, backbone_learning_rate=1e-4, weight_decay=1e-4
        )
