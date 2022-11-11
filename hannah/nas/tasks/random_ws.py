import torch

from .base import SearchTask

from pytorch_lightning import LightningModule
from hydra.utils import instantiate, get_class
from hannah.nas.search_space.parameter_manager import BaseParameterManager

from copy import deepcopy


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
        input_size = [1] + train_set.size()
        self.search_space.prepare(input_size)
        self.wsp = BaseParameterManager(self.search_space, input_size)
        trained_weights = {}
        for i in range(self.budget):
            random_cfg = self.search_space.get_random_cfg(self.search_space.get_config_dims())
            model = self.search_space.sample(random_cfg)
            self.wsp.load_weights(model)
            untrained_weights = deepcopy(model.state_dict())
            if i > 0:
                print("WEIGHTS EQUAL BEFORE TRAINING: {}".format(all([torch.equal(wb, wn) for (k, wb), (k_, wn) in zip(trained_weights.items(), untrained_weights.items())])))
            example_input_array = torch.rand(input_size)
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
            print("WEIGHTS EQUAL AFTER TRAINING: {}".format(all([torch.equal(wb, wn) for (k, wb), (k_, wn) in zip(self.wsp.shared_weights.items(), module.model.state_dict().items())])))
            self.wsp.save_weights(module.model)
