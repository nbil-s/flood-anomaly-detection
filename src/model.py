
import torch
from torch import nn


class LSTMAutoencoder(nn.Module):
    """
    LSTM Autoencoder for multivariate time-series reconstruction.

    Input shape:
        (batch_size, sequence_length, input_features)

    Output shape:
        (batch_size, sequence_length, input_features)
    """

    def __init__(
        self,
        input_features=4,
        hidden_size=32,
        latent_size=16,
    ):
        super().__init__()

        self.input_features = input_features
        self.hidden_size = hidden_size
        self.latent_size = latent_size

        # Encoder: summarizes the input sequence.
        self.encoder = nn.LSTM(
            input_size=input_features,
            hidden_size=hidden_size,
            batch_first=True,
        )

        # Compress the encoder representation.
        self.to_latent = nn.Linear(
            hidden_size, latent_size
        )

        # Expand the latent representation.
        self.from_latent = nn.Linear(
            latent_size, hidden_size
        )

        # Decoder: reconstructs the sequence.
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True,
        )

        # Reconstruct the four sensor measurements.
        self.output_layer = nn.Linear(
            hidden_size, input_features
        )

    def forward(self, x):
        # x shape: (batch, sequence, features)
        _, (hidden, _) = self.encoder(x)

        # Use the final encoder hidden state.
        context = hidden[-1]

        # Compress and expand the context.
        latent = self.to_latent(context)
        expanded = self.from_latent(latent)

        # Repeat the context across all time steps.
        sequence_length = x.size(1)
        repeated = expanded.unsqueeze(1).repeat(
            1, sequence_length, 1
        )

        # Reconstruct the original sequence.
        decoded, _ = self.decoder(repeated)
        reconstruction = self.output_layer(decoded)

        return reconstruction


if __name__ == "__main__":
    # Test the model architecture before training.
    model = LSTMAutoencoder(
        input_features=4,
        hidden_size=32,
        latent_size=16,
    )

    sample_input = torch.randn(8, 12, 4)
    output = model(sample_input)

    print("Model created successfully.")
    print("Input shape:", tuple(sample_input.shape))
    print("Output shape:", tuple(output.shape))
    print(
        "Trainable parameters:",
        sum(p.numel() for p in model.parameters()
            if p.requires_grad),
    )

    assert output.shape == sample_input.shape
    print("Architecture test passed.")
