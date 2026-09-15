"""Assignment 1 model interfaces."""

from src.models.base import ImageClassifier, build_model
from src.models.cnn import CNNClassifier
from src.models.gru import GRUClassifier
from src.models.linear import LinearClassifier
from src.models.mlp import MLPClassifier
from src.models.recurrent import LSTMClassifier
from src.models.transformer import TransformerClassifier

__all__ = [
    "CNNClassifier",
    "GRUClassifier",
    "ImageClassifier",
    "LSTMClassifier",
    "LinearClassifier",
    "MLPClassifier",
    "TransformerClassifier",
    "build_model",
]
