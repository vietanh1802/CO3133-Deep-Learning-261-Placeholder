"""Loss choices for the controlled SUIM loss comparison."""

from torch import Tensor, nn
from torch.nn import functional


class CrossEntropyDiceLoss(nn.Module):
    """Equal-weight CE and per-image macro soft Dice, including all classes.

    Dice uses epsilon 1e-6 and averages over classes and images, including absent
    classes. Per-image reduction keeps its definition independent of accumulation.
    """

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        probabilities = logits.softmax(dim=1)
        labels = functional.one_hot(targets, num_classes=logits.shape[1]).permute(0, 3, 1, 2)
        intersection = (probabilities * labels).sum(dim=(2, 3))
        total = probabilities.sum(dim=(2, 3)) + labels.sum(dim=(2, 3))
        dice_loss = 1 - ((2 * intersection + 1e-6) / (total + 1e-6)).mean()
        return 0.5 * functional.cross_entropy(logits, targets) + 0.5 * dice_loss


def build_segmentation_loss(name: str) -> nn.Module:
    """Construct the configured loss without silently accepting an unknown name."""
    if name == "cross_entropy":
        return nn.CrossEntropyLoss()
    if name == "ce_dice":
        return CrossEntropyDiceLoss()
    raise ValueError("loss must be cross_entropy or ce_dice")
