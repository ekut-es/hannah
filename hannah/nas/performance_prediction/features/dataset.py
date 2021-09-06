import dgl
from dgl.data import DGLDataset
from pathlib import Path

from hannah.nas.graph_conversion import model_to_graph
from hannah.nas.performance_prediction.simple import to_dgl_graph
import torch
import pandas as pd
import numpy as np
import yaml
from hydra.utils import instantiate
import hydra
from omegaconf import OmegaConf, DictConfig

import hannah.conf


class NASGraphDataset(DGLDataset):
    def __init__(self, result_file_path):
        self.result_file_path = result_file_path
        super().__init__(name="nasgraph")

    def process(self):
        self.graphs = []
        self.labels = []
        result_path = Path(self.result_file_path)
        result_file = result_path / "results.yaml"
        assert result_file.exists()
        with result_file.open("r") as result_file:
            results = yaml.safe_load(result_file)

        for i, result in enumerate(results):
            print("Processing model {}".format(i))
            config_file_path = result_path / result["config"]
            with config_file_path.open("r") as config_file:
                config = yaml.safe_load(config_file)
            # config = DictConfig(config)
            # hydra.initialize_config_module('hannah.conf')
            # config = hydra.compose("config")
            config = OmegaConf.create(config)
            metrics = result["metrics"]
            # backend = instantiate(config.backend)
            model = instantiate(
                config.module,
                dataset=config.dataset,
                model=config.model,
                optimizer=config.optimizer,
                features=config.features,
                scheduler=config.get("scheduler", None),
                normalizer=config.get("normalizer", None),
                _recursive_=False,
            )
            model.setup("test")

            dgl_graph = to_dgl_graph(model_to_graph(model))
            self.graphs.append(dgl_graph)
            label = metrics["val_error"]  # 1 / metrics['val_error']
            self.labels.append(label)

        self.labels = torch.FloatTensor(self.labels)

    def normalize_labels(self):
        std = self.labels.std()
        mean = self.labels.mean()
        self.labels = (self.labels - mean) / std

    def normalize_features(self, max_feature):
        for g in self.graphs:
            new_features = g.ndata["features"].clone()
            for row in range(len(new_features)):
                new_features[row] = new_features[row] / max_feature
            g.ndata["features"] = new_features

    def to_class_labels(self):
        new_labels = []
        for l in self.labels:
            if l > 0:
                new_labels.append(1)
            else:
                new_labels.append(0)
        self.float_labels = self.labels.clone()
        self.labels = torch.LongTensor(new_labels)

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)


@hydra.main(config_path="../../../conf", config_name="config")
def main(config):
    dataset = NASGraphDataset(
        "/local/gerum/speech_recognition/characterize/nas_kws2/conv_net_trax/n1sdp/"
    )


if __name__ == "__main__":
    main()
