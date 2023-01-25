#
# Copyright (c) 2023 Hannah contributors.
#
# This file is part of hannah.
# See https://github.com/ekut-es/hannah for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import statistics

import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import Callback
from tqdm import trange


class LinearClassifierTraining(Callback):
    def __init__(self, epochs, learning_rate):
        self.epochs = epochs
        self.learning_rate = learning_rate
        super().__init__()

    def on_train_end(self, trainer, pl_module):
        if (
            pl_module.hparams.train_val_loss == "decoder"
            and pl_module.model.classifier is not None
        ):
            optimizer = torch.optim.AdamW(
                pl_module.model.classifier.parameters(), lr=self.learning_rate
            )
            print("Starting training of linear classifier.")
            for epoch in trange(self.epochs):
                losses = []
                for batch in trainer.train_dataloader:
                    labeled_batch = batch["labeled"]
                    x = labeled_batch["data"]
                    labels = labeled_batch.get("labels", None).to(
                        device=pl_module.device
                    )
                    boxes = labeled_batch.get("bbox", [])
                    augmented_data, x = pl_module.augment(x, labels, boxes, 1)
                    augmented_data = augmented_data.to(device=pl_module.device)
                    prediction_result = pl_module.model.forward(augmented_data)
                    optimizer.zero_grad()
                    logits = pl_module.model.classifier(prediction_result.latent)
                    current_loss = F.cross_entropy(logits, labels)
                    preds = torch.argmax(logits, dim=1)
                    current_loss.backward()
                    optimizer.step()
                    losses.append(current_loss)

                trainer.logger.log_metrics(
                    {"linear_classifier_train_loss": statistics.fmean(losses)},
                    step=epoch,
                )
        else:
            return super().on_train_end(trainer, pl_module)
        return super().on_train_end(trainer, pl_module)
