"""
Unit tests for the LSTM model.
No GPU, no MLflow, no DB — pure PyTorch.
"""

import torch


def test_model_output_shape():
    """Model should output (batch, 1) for any valid batch size."""
    import sys

    sys.path.insert(0, "services/training/src")
    from training.models.lstm_transformer import LSTMTransformerModel

    model = LSTMTransformerModel()
    model.eval()

    for batch_size in [1, 8, 32]:
        x = torch.randn(batch_size, 60, 25)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (batch_size, 1), f"Bad shape for batch={batch_size}: {out.shape}"


def test_model_output_in_range():
    """Output must be a probability in [0, 1] (Sigmoid applied)."""
    import sys

    sys.path.insert(0, "services/training/src")
    from training.models.lstm_transformer import LSTMTransformerModel

    model = LSTMTransformerModel()
    model.eval()
    x = torch.randn(16, 60, 25)
    with torch.no_grad():
        out = model(x)
    assert out.min() >= 0.0, "Output below 0"
    assert out.max() <= 1.0, "Output above 1"


def test_model_deterministic_in_eval():
    """In eval mode, same input should always give same output."""
    import sys

    sys.path.insert(0, "services/training/src")
    from training.models.lstm_transformer import LSTMTransformerModel

    model = LSTMTransformerModel()
    model.eval()
    x = torch.randn(4, 60, 25)
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)
    assert torch.allclose(out1, out2), "Non-deterministic in eval mode"
