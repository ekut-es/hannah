import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def load_model(path: str, **args):
    model_path = Path(path)
    logger.info("Loading pickled model from: %s", str(path))

    with model_path.open("rb") as f:
        model = torch.load(f, map_location="cpu")

    logger.info("Loaded model:\n%s", str(model))

    return model
