"""Raw dataset contract used by exploratory analysis."""

from typing import Any, Protocol


class RawImageDataset(Protocol):
    """The part of a torchvision dataset the EDA needs: undecoded images and labels."""

    data: Any
    targets: Any
