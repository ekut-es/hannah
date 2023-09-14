import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from hannah.nas.performance_prediction.features.dataset import NASGraphDataset
from hannah.nas.performance_prediction.gcn.predictor import (
    GCNPredictor,
    GaussianProcessPredictor,
    XGBPredictor,
    get_input_feature_size,
    prepare_dataloader,
)
from sklearn.gaussian_process.kernels import (
    RBF,
    DotProduct,
    Kernel,
    Matern,
    RationalQuadratic,
    WhiteKernel,
)

if __name__ == '__main__':
    dataset_path = "/home/moritz/projects/hannah/experiments/trained_models/ri_capsule_baseline/lazy_resnet/performance_data"
    dataset = NASGraphDataset(dataset_path)
    dataset.normalize_features()

    train_dataloader, test_dataloader = prepare_dataloader(
            dataset, batch_size=20, train_test_split=0.8
        )
    in_feat = get_input_feature_size(dataset)
    # predictor = GaussianProcessPredictor(
    #     in_feat, hidden_units=[8], embedding_size=10, readout="mean", kernel=RBF() + DotProduct(), alpha=1e-3
    # )
    # predictor = GCNPredictor(in_feat, hidden_units=[32, 32, 32])
    xgb_param = {
                "max_depth": 20,
                "eta": 0.1,
                "gamma": 0.01,
                "objective": "reg:squarederror",
            }
    predictor = XGBPredictor(input_feature_size=in_feat, hidden_units=[8], embedding_size=10 ,xgb_param=xgb_param)
    predictor.train_and_fit(
        train_dataloader,
        num_epochs=400,
        learning_rate=0.001,
        validation_dataloader=test_dataloader,
        verbose=25,
    )

    pred_train = pd.DataFrame(columns=["pred", "real"])
    total_loss = 0
    num_tests = 0
    for batched_graph, labels in train_dataloader:
        preds = predictor.predict(batched_graph)
        if isinstance(preds, tuple):
            pred, std = predictor.predict(batched_graph)
        else:
            pred = predictor.predict(batched_graph)
        for p, l in zip(pred, labels):
            pred_train = pred_train.append(
                {"pred": p.item(), "real": l.item()}, ignore_index=True
            )

        loss = F.mse_loss(torch.tensor(pred), labels, reduction="mean").item()
        total_loss += loss
        num_tests += len(labels)
    print("MSE (Train): {}".format(total_loss / num_tests))

    fig = plt.figure(figsize=(20, 8))
    d = pred_train.sort_values(by="real")
    x = np.arange(len(d))
    plt.scatter(x, d["pred"], label="Prediction")
    plt.scatter(x, d["real"], label="Real Value")
    plt.legend()
    plt.ylabel("1 / cost (time in ms)")
    # plt.savefig('../experiments/performance_prediction/prediction_acc_h50_b250_e400')
    plt.show()

    pred_eval = pd.DataFrame(columns=["pred", "real"])
    total_loss = 0
    num_tests = 0
    for batched_graph, labels in test_dataloader:
        preds = predictor.predict(batched_graph)
        if isinstance(preds, tuple):
            pred, std = predictor.predict(batched_graph)
        else:
            pred = predictor.predict(batched_graph)
        for p, l in zip(pred, labels):
            pred_eval = pred_eval.append(
                {"pred": p.item(), "real": l.item()}, ignore_index=True
            )

        loss = F.mse_loss(torch.tensor(pred), labels, reduction="mean").item()
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
    print()
