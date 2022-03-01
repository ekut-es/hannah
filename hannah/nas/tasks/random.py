import torch

from .base import SearchTask

from pytorch_lightning import LightningModule
from hydra.utils import instantiate, get_class
from hannah.nas.search_space.utils import get_random_cfg


class RandomSearchTask(SearchTask):
    def __init__(self, budget=2000, *args, **kwargs):
        super().__init__(*args, budget=budget, **kwargs)

    def fit(self, module: LightningModule):
        trainer = instantiate(self.config.trainer)
        trainer.fit(module)

    def run(self):

        # Prepare dataset
        get_class(self.config.dataset.cls).prepare(self.config.dataset)

        # Instantiate Dataset
        train_set, val_set, test_set = get_class(self.config.dataset.cls).splits(
            self.config.dataset
        )

        self.search_space.prepare([1] + train_set.size())
        for i in range(self.budget):
            model = self.search_space.sample(config=get_random_cfg(self.search_space.get_config_dims()))
            example_input_array = torch.rand([1] + train_set.size())
            module = instantiate(
                self.config.module,
                model=model,
                dataset=self.config.dataset,
                optimizer=self.config.optimizer,
                features=self.config.features,
                normalizer=self.config.get("normalizer", None),
                scheduler=self.config.scheduler,
                example_input_array=example_input_array,
                num_classes=len(train_set.class_names),
                _recursive_=False,
            )
            self.fit(module)
