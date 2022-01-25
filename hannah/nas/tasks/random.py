import torch
import pytorch_lightning

from .base import SearchTask

from pytorch_lightning import LightningModule
from hydra.utils import instantiate, get_class

from hannah.datasets.base import ctc_collate_fn


class RandomSearchTask(SearchTask):
    def __init__(self, budget=2000, *args, **kwargs):
        super().__init__(*args, budget=budget, **kwargs)

    def fit(self, module: LightningModule, train_set, val_set):
        trainer = pytorch_lightning.Trainer(gpus=0, max_steps=100)

        trainer.fit(
            module,
            train_dataloaders=torch.utils.data.DataLoader(train_set, batch_size=32),
            val_dataloaders=[torch.utils.data.DataLoader(val_set, batch_size=32)],
        )

    def run(self):

        # Prepare dataset
        get_class(self.config.dataset.cls).prepare(self.config.dataset)

        # Instantiate Dataset
        train_set, val_set, test_set = get_class(self.config.dataset.cls).splits(
            self.config.dataset
        )

        for i in range(self.budget):
            model = self.search_space.sample([1] + train_set.size())
            example_input_array = torch.rand([1] + train_set.size())
            module = instantiate(
                self.config.module,
                model=model,
                optimizer=self.config.optimizer,
                features=self.config.features,
                normalizer=self.config.get("normalizer", None),
                scheduler=self.config.scheduler,
                example_input_array=example_input_array,
                num_classes=len(train_set.class_names),
                _recursive_=False,
            )
            self.fit(module, train_set, val_set)
