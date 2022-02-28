import torch.nn as nn


class LSTMModel(nn.Module):
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
