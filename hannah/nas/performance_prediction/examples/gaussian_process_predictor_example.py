from hannah.nas.performance_prediction.gcn.predictor import (
    GaussianProcessPredictor,
    prepare_dataloader,
    get_input_feature_size,
)
from hannah.nas.performance_prediction.features.dataset import NASGraphDataset

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt
import hydra

plt.style.use("seaborn")


@hydra.main(config_path="../../../conf", config_name="config")
def main(config):
    # dataset = NASGraphDataset(cfg_space, edge_file, prop_file)
    # dataset = NASGraphDataset('/local/gerum/speech_recognition/characterize/nas_kws2/conv_net_trax/n1sdp/')
    dataset = NASGraphDataset(
        "/local/gerum/speech_recognition/trained_models/nas_kws2/conv_net_trax/results"
    )

    train_dataloader, test_dataloader = prepare_dataloader(
        dataset, batch_size=250, train_test_split=0.7
    )
    in_feat = get_input_feature_size(dataset)
    predictor = GaussianProcessPredictor(
        in_feat, hidden_units=[512], embedding_size=10, readout="mean", kernel="default"
    )

    predictor.train_and_fit(
        train_dataloader,
        num_epochs=20,
        learning_rate=0.001,
        validation_dataloader=test_dataloader,
        verbose=25,
    )

    pred_eval = pd.DataFrame(columns=["pred", "real"])
    total_loss = 0
    num_tests = 0
    for batched_graph, labels in test_dataloader:
        pred, std = predictor.predict(batched_graph)
        for p, l in zip(pred, labels):
            pred_eval = pred_eval.append(
                {"pred": p, "real": l.item()}, ignore_index=True
            )

        loss = F.mse_loss(torch.tensor(pred), labels, reduction="sum").item()
        total_loss += loss
        num_tests += len(labels)
    print("MSE (Test): {}".format(total_loss / num_tests))

    fig = plt.figure(figsize=(20, 8))
    d = pred_eval.sort_values(by="real")
    x = np.arange(len(d))
    plt.scatter(x, d["pred"], label="Prediction")
    plt.scatter(x, d["real"], label="Real Value")
    plt.legend()
    plt.ylabel("1 / cost (time in ms)")
    # plt.savefig('../experiments/performance_prediction/prediction_acc_h50_b250_e400')
    plt.show()


if __name__ == "__main__":
    main()
