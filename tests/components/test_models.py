from typing import cast

import pytest
import torch
import torch.nn as nn

from src.models import GRU, MLP, ImageClassifier, Linear, Transformer, build_model
from src.models.gru import GRUCell


def test_model_factory_dispatches_to_named_model() -> None:
    model = build_model("linear", {})

    assert isinstance(model, Linear)


def test_model_factory_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unknown model"):
        build_model("unknown", {})


def test_gru_cell_matches_gate_equations() -> None:
    torch.manual_seed(0)
    cell = GRUCell(input_size=3, hidden_size=5)
    inputs = torch.randn(2, 3)
    hidden = torch.randn(2, 5)

    projected_input = cell.project_input(inputs)
    projected_hidden = cell.project_hidden(hidden)
    input_reset, input_update, input_candidate = projected_input.chunk(3, dim=-1)
    hidden_reset, hidden_update, hidden_candidate = projected_hidden.chunk(3, dim=-1)
    reset = torch.sigmoid(input_reset + hidden_reset)
    update = torch.sigmoid(input_update + hidden_update)
    candidate = torch.tanh(input_candidate + reset * hidden_candidate)
    expected = (1 - update) * candidate + update * hidden

    torch.testing.assert_close(cell(inputs, hidden), expected)


def test_gru_returns_final_features_and_gradients() -> None:
    model = GRU(input_size=4, hidden_size=6, num_layers=2, dropout=0.1)
    images = torch.randn(3, 1, 5, 4)

    features = model(images)
    features.sum().backward()

    assert features.shape == (3, 6)
    assert model.output_dim == 6
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_gru_accepts_rgb_image_rows() -> None:
    model = GRU(input_size=12, hidden_size=6)

    features = model(torch.randn(2, 3, 5, 4))

    assert features.shape == (2, 6)


def test_gru_rejects_invalid_image_shape() -> None:
    model = GRU(input_size=4, hidden_size=6)

    with pytest.raises(ValueError, match="images must have shape"):
        model(torch.randn(3, 4, 4))


def test_gru_matches_nn_gru() -> None:
    our_model = GRU(input_size=4, hidden_size=6, num_layers=2, dropout=0.0)
    nn_model = nn.GRU(input_size=4, hidden_size=6, num_layers=2, dropout=0.0, batch_first=True)

    with torch.no_grad():
        for layer_index, module in enumerate(our_model.cells):
            cell = cast(GRUCell, module)
            assert cell.project_input.bias is not None
            assert cell.project_hidden.bias is not None
            getattr(nn_model, f"weight_ih_l{layer_index}").copy_(cell.project_input.weight)
            getattr(nn_model, f"bias_ih_l{layer_index}").copy_(cell.project_input.bias)
            getattr(nn_model, f"weight_hh_l{layer_index}").copy_(cell.project_hidden.weight)
            getattr(nn_model, f"bias_hh_l{layer_index}").copy_(cell.project_hidden.bias)

    images = torch.randn(3, 1, 5, 4)
    sequence = images.squeeze(1)

    our_features = our_model(images)
    _, final_hidden = nn_model(sequence)
    nn_features = final_hidden[-1]

    torch.testing.assert_close(our_features, nn_features)


def test_mlp_classifier_produces_logits_and_gradients() -> None:
    model = ImageClassifier(MLP(input_dim=16, hidden_dims=(8,), dropout=0.1), num_classes=3)
    images = torch.randn(5, 1, 4, 4)
    targets = torch.tensor([0, 1, 2, 1, 0])

    logits = model(images)
    loss = nn.CrossEntropyLoss()(logits, targets)
    loss.backward()

    assert logits.shape == (5, 3)
    assert torch.isfinite(loss)
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_transformer_produces_fixed_size_features() -> None:
    model = Transformer(token_dim=4, embedding_dim=8, num_heads=2, num_layers=1)

    features = model(torch.randn(3, 1, 5, 4))

    assert features.shape == (3, 8)
    assert model.output_dim == 8


def test_transformer_accepts_rgb_image_rows() -> None:
    model = Transformer(token_dim=12, embedding_dim=8, num_heads=2, num_layers=1)

    features = model(torch.randn(2, 3, 5, 4))

    assert features.shape == (2, 8)
