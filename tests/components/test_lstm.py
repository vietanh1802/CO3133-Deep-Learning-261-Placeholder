from pathlib import Path

import pytest
import torch
from torch import nn

from src.config.classification import load_config
from src.models import ImageClassifier, build_model
from src.models.lstm import LSTM, LSTMCell

CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "configs/image_classification/fashion_mnist/lstm.yaml"
)


@pytest.mark.parametrize("layers", [1, 2, 3])
@pytest.mark.parametrize("batch,length", [(1, 1), (3, 7)])
@pytest.mark.parametrize("channels", [1, 3])
def test_lstm_matches_reference_outputs_and_gradients(layers, batch, length, channels):
    torch.manual_seed(42)
    model = LSTM(input_size=channels * 4, hidden_size=5, num_layers=layers).double()
    reference = nn.LSTM(channels * 4, 5, num_layers=layers, batch_first=True).double()
    with torch.no_grad():
        for index, cell in enumerate(model.cells):
            assert isinstance(cell, LSTMCell)
            assert cell.project_input.bias is not None
            assert cell.project_hidden.bias is not None
            for name, parameter in (
                ("weight_ih", cell.project_input.weight),
                ("bias_ih", cell.project_input.bias),
                ("weight_hh", cell.project_hidden.weight),
                ("bias_hh", cell.project_hidden.bias),
            ):
                getattr(reference, f"{name}_l{index}").copy_(parameter)

    images = torch.randn(batch, channels, length, 4, dtype=torch.float64, requires_grad=True)
    sequence = (
        images.detach().permute(0, 2, 1, 3).reshape(batch, length, channels * 4).clone()
    ).requires_grad_(True)
    features = model(images)
    _, (hidden, _) = reference(sequence)
    torch.testing.assert_close(features, hidden[-1], rtol=1e-9, atol=1e-11)

    probe = torch.randn_like(features)
    (features * probe).sum().backward()
    (hidden[-1] * probe).sum().backward()
    assert images.grad is not None
    assert sequence.grad is not None
    image_grad = images.grad.permute(0, 2, 1, 3).reshape(batch, length, channels * 4)
    torch.testing.assert_close(image_grad, sequence.grad, rtol=1e-9, atol=1e-11)
    for index, cell in enumerate(model.cells):
        assert isinstance(cell, LSTMCell)
        assert cell.project_input.bias is not None
        assert cell.project_hidden.bias is not None
        for name, parameter in (
            ("weight_ih", cell.project_input.weight),
            ("bias_ih", cell.project_input.bias),
            ("weight_hh", cell.project_hidden.weight),
            ("bias_hh", cell.project_hidden.bias),
        ):
            torch.testing.assert_close(
                parameter.grad,
                getattr(reference, f"{name}_l{index}").grad,
                rtol=1e-9,
                atol=1e-11,
            )


def test_lstm_config_builds_fashion_mnist_classifier():
    config = load_config(CONFIG_PATH)
    encoder = build_model(config.model, config.model_config)
    assert isinstance(encoder, LSTM)
    model = ImageClassifier(encoder)
    logits = model(torch.randn(3, 1, 28, 28))
    loss = nn.CrossEntropyLoss()(logits, torch.tensor([0, 4, 9]))
    loss.backward()
    assert logits.shape == (3, 10)
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize("layers", [1, 2])
def test_lstm_dropout_modes_and_no_state_leak_between_batches(layers):
    torch.manual_seed(17)
    model = LSTM(input_size=4, hidden_size=8, num_layers=layers, dropout=0.5)
    images = torch.randn(3, 1, 6, 4)
    model.eval()
    expected = model(images)
    model(torch.randn_like(images))
    torch.testing.assert_close(model(images), expected, rtol=0, atol=0)
    model.train()
    first, second = model(images), model(images)
    if layers == 1:
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    else:
        assert not torch.equal(first, second)


@pytest.mark.parametrize("shape", [(2, 3, 5, 4), (2, 5, 4), (2, 1, 5, 3), (2, 1, 0, 4)])
def test_lstm_rejects_invalid_image_shapes(shape):
    with pytest.raises(ValueError):
        LSTM(input_size=4)(torch.randn(*shape))


def test_lstm_cell_rejects_mismatched_state():
    with pytest.raises(ValueError, match="hidden and cell must have the same shape"):
        LSTMCell(3, 5)(torch.randn(2, 3), torch.randn(2, 5), torch.randn(2, 4))
