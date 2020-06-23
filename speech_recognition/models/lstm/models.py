import torch.nn as nn

from ..utils import ConfigType, SerializableModule


class LSTMModel(SerializableModule):
    """Simple LSTM model."""

    def __init__(self, config):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=config["height"],
            hidden_size=config["hidden_size"],
            num_layers=config["n_layers"],
            batch_first=True,
            dropout=config["dropout"],
        )
        self.dropout = nn.Dropout(config["dropout"])
        self.fc = nn.Linear(config["hidden_size"], config["n_labels"])

    def forward(self, x):
        x = x.permute(0, 2, 1)
        _, (ht, _) = self.lstm(x)
        x = self.dropout(ht[-1])
        x = self.fc(x)
        return x


configs = {
    ConfigType.LSTM_1.value: dict(
        features="mel",
        hidden_size=64,
        n_layers=1,
        dropout=0.2,
        lr=0.001,
        optimizer="adam",
        freq_min=0,
        freq_max=8000,
        n_mels=32,
        n_mfcc=13,
    )
}
