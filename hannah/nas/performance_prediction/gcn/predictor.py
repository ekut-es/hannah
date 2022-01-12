from .model import GCN, GCNEmbedding
import torch
import dgl
import torch.nn.functional as F
import numpy as np
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import xgboost as xgb

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    DotProduct,
    WhiteKernel,
    RBF,
    RationalQuadratic,
    Matern,
    Kernel,
)


class Predictor:
    def __init__(self, fea_name="features") -> None:
        """Parent method for different predictor classes.

        Parameters
        ----------
        fea_name : str, optional
            internal name for features in the graph, as in graph.ndata[fea_name], by default 'features'
        """

        self.fea_name = fea_name
        self.model = None

    def train(
        self,
        dataloader,
        learning_rate=1e-3,
        num_epochs=200,
        validation_dataloader=None,
        verbose=1,
    ):
        """Train GCN model

        Parameters
        ----------
        dataloader : GraphDataLoader
            training data
        learning_rate : [type], optional
            by default 1e-3
        num_epochs : int, optional
            by default 200
        validation_dataloader : [type], optional
            if given, use this data to print validation loss, by default None
        verbose : int
            if validation_dataloader is given, print validation MSE every <verbose> epoch, by default 1
        """
        assert self.model, "You must specify a model (e.g. use GCNPredictor())"
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        for epoch in range(num_epochs):
            for batched_graph, labels in dataloader:
                pred = self.model(
                    batched_graph, batched_graph.ndata[self.fea_name].float()
                ).squeeze()
                loss = F.mse_loss(pred, labels, reduction="sum")
                # loss = F.l1_loss(pred, labels, reduction="sum")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if validation_dataloader:
                total_loss = 0
                num_tests = 0
                for vbatched_graph, vlabels in validation_dataloader:
                    vpred = self.model(
                        vbatched_graph, vbatched_graph.ndata[self.fea_name].float()
                    ).squeeze()
                    vloss = F.mse_loss(vpred, vlabels, reduction="sum").item()
                    # vloss = F.l1_loss(pred, labels, reduction="sum")
                    total_loss += vloss
                    num_tests += len(vlabels)
                if (verbose and epoch % verbose == 0) or epoch == num_epochs - 1:
                    print(
                        "Epoch {} Validation MSE: {:.5f}".format(
                            epoch, total_loss / num_tests
                        )
                    )

    def predict(self, graph):
        """predict cost of graph

        Parameters
        ----------
        graph : dgl.Graph

        Returns
        -------
        torch.Tensor
            predicted cost of given graph. Retrieve float value with .item()
        """
        assert self.model, "You must specify a model (e.g. use GCNPredictor())"
        pred = self.model(graph, graph.ndata[self.fea_name].float())
        return pred


class GCNPredictor(Predictor):
    def __init__(
        self, input_feature_size, hidden_units=128, readout="mean", fea_name="features"
    ) -> None:
        """G(raph)CN based network latency/cost predictor. End-to-end from graph to score.

        Parameters
        ----------
        input_feature_size : [type]
            length of feature vector of a graph node (graph G with n nodes, each with features of length m, i.e. feature matrix F = n x m)
        hidden_units : int, list, optional
            size of hidden layer (layers if list) , by default 128
        readout : str, optional
            readout function that is used to aggregate node features, by default 'mean'
        fea_name : str, optional
            internal name for features in the graph, as in graph.ndata[fea_name], by default 'features'
        """
        super().__init__(fea_name)
        self.model = GCN(
            input_feature_size, hidden_units, num_classes=1, readout=readout
        )

    def train(
        self,
        dataloader,
        learning_rate=1e-3,
        num_epochs=200,
        validation_dataloader=None,
        verbose=0,
    ):
        """Train GCN model

        Parameters
        ----------
        dataloader : GraphDataLoader
            training data
        learning_rate : [type], optional
            by default 1e-3
        num_epochs : int, optional
            by default 200
        validation_dataloader : [type], optional
            if given, use this data to print validation loss, by default None
        verbose : int
            if validation_dataloader is given, print validation MSE every <verbose> epoch, by default 1
        """
        super().train(
            dataloader, learning_rate, num_epochs, validation_dataloader, verbose
        )

    def predict(self, graph):
        """predict cost of graph

        Parameters
        ----------
        graph : dgl.Graph

        Returns
        -------
        torch.Tensor
            predicted cost of given graph. Retrieve float value with .item()
        """
        return super().predict(graph)


class GaussianProcessPredictor(Predictor):
    def __init__(
        self,
        input_feature_size,
        hidden_units=128,
        embedding_size=10,
        readout="mean",
        fea_name="features",
        kernel="default",
        alpha=1e-10
    ) -> None:
        """Predictor that generates a graph embedding that is used as input for a gaussian process predictor.

        Parameters
        ----------
        input_feature_size : [type]
            length of feature vector of a graph node (graph G with n nodes, each with features of length m, i.e. feature matrix F = n x m)
        hidden_units : int, list, optional
            size of hidden layer (layers if list) , by default 128
        embedding_size: int, optional
            size of output embedding
        readout : str, optional
            readout function that is used to aggregate node features, by default 'mean'
        fea_name : str, optional
            internal name for features in the graph, as in graph.ndata[fea_name], by default 'features'
        kernel : str, sklearn.gaussian_process.kernels.Kernel, optional
            The gaussian process kernel to use.
            input shoudl be either "default", or a sklearn Kernel() object
            by default RBF() + DotProduct() + WhiteKernel()
        """
        super().__init__(fea_name)
        if isinstance(hidden_units, int):
            hidden_units = [hidden_units]
        self.model = GCNEmbedding(
            input_feature_size,
            hidden_units,
            embedding_size=embedding_size,
            readout=readout,
        )
        if kernel == "default":
            kernel = RBF() + DotProduct() + WhiteKernel()
        else:
            assert isinstance(kernel, Kernel), "Not a valid kernel."
        self.predictor = GaussianProcessRegressor(kernel=kernel, random_state=0, normalize_y=True, n_restarts_optimizer=2, alpha=alpha)

    def set_predictor(self, predictor):
        self.predictor = predictor

    def set_embedding_model(self, model):
        self.model = model

    def fit_predictor(self, embeddings, labels):
        self.predictor.fit(embeddings, labels)

    def embedd_and_fit(self, dataloader, verbose=True):
        embeddings = []
        labels = []
        for batched_graph, batched_labels in dataloader:
            graphs = dgl.unbatch(batched_graph)
            for g, l in zip(graphs, batched_labels):
                embeddings.append(self.get_embedding(g))
                labels.append(l)

        embeddings = torch.vstack(embeddings).detach().numpy()
        labels = torch.hstack(labels).detach().numpy()

        self.fit_predictor(embeddings, labels)
        score = self.predictor.score(embeddings, labels)
        if verbose:
            print("Predictor Score: {:.5f}".format(score))
        return score

    def embedd(self, dataloader):
        embeddings = []
        labels = []
        for batched_graph, batched_labels in dataloader:
            graphs = dgl.unbatch(batched_graph)
            for g, l in zip(graphs, batched_labels):
                embeddings.append(self.get_embedding(g))
                labels.append(l)

        embeddings = torch.vstack(embeddings).detach().numpy()
        labels = torch.hstack(labels).detach().numpy()
        return embeddings, labels



    def train_and_fit(
        self,
        dataloader,
        learning_rate=1e-3,
        num_epochs=200,
        validation_dataloader=None,
        verbose=1,
    ):
        """Train GCN model, generate embeddings for training data and fit the predictor with embeddings.

        Parameters
        ----------
        dataloader : GraphDataLoader
            training data
        learning_rate : [type], optional
            by default 1e-3
        num_epochs : int, optional
            by default 200
        validation_dataloader : [type], optional
            if given, use this data to print validation loss, by default None
        verbose : int
            if validation_dataloader is given, print validation MSE every <verbose> epoch,by default 1

        Returns
        -------
        float
            score of predictor on TRAINING data, see sklearn doc of chosen predictor for more info
        """
        if verbose:
            print("Train embedding network ...")
        super().train(
            dataloader, learning_rate, num_epochs, validation_dataloader, verbose
        )

        if verbose:
            print("Create training embeddings ...")

        embeddings = []
        labels = []
        for batched_graph, batched_labels in dataloader:
            graphs = dgl.unbatch(batched_graph)
            for g, l in zip(graphs, batched_labels):
                embeddings.append(self.get_embedding(g))
                labels.append(l)

        embeddings = torch.vstack(embeddings).detach().numpy()
        labels = torch.hstack(labels).detach().numpy()

        if verbose:
            print("Fit predictor ...")

        self.fit_predictor(embeddings, labels)
        score = self.predictor.score(embeddings, labels)
        if verbose:
            print("Predictor Score: {:.5f}".format(score))
        return score

    def get_embedding(self, graph):
        return self.model.get_embedding(graph, graph.ndata[self.fea_name].float())

    def score(self, X, y):
        pass

    def predict(self, X, return_std=True):
        """Predict cost/latency of graphs.

        Parameters
        ----------
        X : dgl.DGLGraph, list[DGLGraph], dgl.dataloading.GraphDataLoader
            Input graph(s)
        return_std : bool, optional
            if true, return standard dev. else just mean prediction, by default True

        Returns
        -------
        array (,array)
            prediction(s) , (if return_std: standard deviation(s))
        """
        if isinstance(X, dgl.DGLGraph):
            if X.batch_size == 1:
                embeddings = self.get_embedding(X).detach().numpy()
            else:
                embeddings = []
                graphs = dgl.unbatch(X)
                for g in graphs:
                    embeddings.append(self.get_embedding(g))
                embeddings = torch.vstack(embeddings).detach().numpy()
        elif isinstance(X, list):
            embeddings = []
            for graph in X:
                embeddings.append(self.get_embedding(graph))
            embeddings = torch.vstack(embeddings).detach().numpy()
        elif isinstance(X, GraphDataLoader):
            embeddings = []
            for batched_graphs, _ in X:
                graphs = dgl.unbatch(batched_graphs)
                for g in graphs:
                    embeddings.append(self.get_embedding(g))

            embeddings = torch.vstack(embeddings).detach().numpy()

        preds = self.predictor.predict(embeddings, return_std=return_std)
        if return_std:
            preds_mean, preds_std = preds
            return preds_mean, preds_std
        else:
            return preds


class XGBPredictor(Predictor):
    def __init__(
        self,
        input_feature_size,
        hidden_units=128,
        embedding_size=10,
        readout="mean",
        fea_name="features",
        xgb_param="default",
    ) -> None:
        """Predictor that generates a graph embedding that is used as input for a xgb based predictor.

        Parameters
        ----------
        input_feature_size : [type]
            length of feature vector of a graph node (graph G with n nodes, each with features of length m, i.e. feature matrix F = n x m)
        hidden_units : int, list, optional
            size of hidden layer (layers if list) , by default 128
        embedding_size: int, optional
            size of output embedding
        readout : str, optional
            readout function that is used to aggregate node features, by default 'mean'
        fea_name : str, optional
            internal name for features in the graph, as in graph.ndata[fea_name], by default 'features'
        xgb_param : str, dict, optional
            The xgb_parameter to use.
            See https://xgboost.readthedocs.io/en/latest/parameter.html
        """
        super().__init__(fea_name)
        if isinstance(hidden_units, int):
            hidden_units = [hidden_units]
        self.model = GCNEmbedding(
            input_feature_size,
            hidden_units,
            embedding_size=embedding_size,
            readout=readout,
        )
        if xgb_param == "default":
            self.xgb_param = {
                "max_depth": 30,
                "eta": 0.5,
                "gamma": 0,
                "objective": "reg:squarederror",
            }
        else:
            self.xgb_param = xgb_param
        self.predictor = None

    def set_predictor(self, predictor):
        self.predictor = predictor

    def set_embedding_model(self, model):
        self.model = model

    def fit_predictor(self, embeddings, labels, num_round=800):
        dtrain = xgb.DMatrix(embeddings, labels)
        self.predictor = xgb.train(self.xgb_param, dtrain, num_round)

    def train_and_fit(
        self,
        dataloader,
        learning_rate=1e-3,
        num_epochs=200,
        num_round=8000,
        validation_dataloader=None,
        verbose=1,
    ):
        """Train GCN model, generate embeddings for training data and fit the predictor with embeddings.

        Parameters
        ----------
        dataloader : GraphDataLoader
            training data
        learning_rate : [type], optional
            by default 1e-3
        num_epochs : int, optional
            Training epochs for the GCN embedding network, by default 200
        num_round : int, optional
            training rounds for xgb booster, by default 800
        validation_dataloader : [type], optional
            if given, use this data to print validation loss, by default None
        verbose : int
            if validation_dataloader is given, print validation MSE every <verbose> epoch,by default 1
        """
        if verbose:
            print("Train embedding network ...")
        super().train(
            dataloader, learning_rate, num_epochs, validation_dataloader, verbose
        )

        if verbose:
            print("Create training embeddings ...")

        embeddings = []
        labels = []
        for batched_graph, batched_labels in dataloader:
            graphs = dgl.unbatch(batched_graph)
            for g, l in zip(graphs, batched_labels):
                embeddings.append(self.get_embedding(g))
                labels.append(l)

        embeddings = torch.vstack(embeddings).detach().numpy()
        labels = torch.hstack(labels).detach().numpy()

        if verbose:
            print("Fit predictor ...")

        self.fit_predictor(embeddings, labels)

    def get_embedding(self, graph):
        return self.model.get_embedding(graph, graph.ndata[self.fea_name].float())

    def score(self, X, y):
        pass

    def predict(self, X):
        """Predict cost/latency of graphs.

        Parameters
        ----------
        X : dgl.DGLGraph, list[DGLGraph], dgl.dataloading.GraphDataLoader
            Input graph(s)
        Returns
        -------
        array (,array)
            prediction(s) , (if return_std: standard deviation(s))
        """
        if isinstance(X, dgl.DGLGraph):
            if X.batch_size == 1:
                embeddings = self.get_embedding(X).detach().numpy()
            else:
                embeddings = []
                graphs = dgl.unbatch(X)
                for g in graphs:
                    embeddings.append(self.get_embedding(g))
                embeddings = torch.vstack(embeddings).detach().numpy()
        elif isinstance(X, list):
            embeddings = []
            for graph in X:
                embeddings.append(self.get_embedding(graph))
            embeddings = torch.vstack(embeddings).detach().numpy()
        elif isinstance(X, GraphDataLoader):
            embeddings = []
            for batched_graphs, _ in X:
                graphs = dgl.unbatch(batched_graphs)
                for g in graphs:
                    embeddings.append(self.get_embedding(g))

            embeddings = torch.vstack(embeddings).detach().numpy()
        dtest = xgb.DMatrix(embeddings)

        preds = self.predictor.predict(dtest)
        return preds


    def embedd_and_fit(self, dataloader, verbose=True):
        embeddings = []
        labels = []
        for batched_graph, batched_labels in dataloader:
            graphs = dgl.unbatch(batched_graph)
            for g, l in zip(graphs, batched_labels):
                embeddings.append(self.get_embedding(g))
                labels.append(l)

        embeddings = torch.vstack(embeddings).detach().numpy()
        labels = torch.hstack(labels).detach().numpy()

        self.fit_predictor(embeddings, labels)


def prepare_dataloader(dataset, batch_size=50, train_test_split=1, subset=0, seed=0, validation=False):
    """ helper function to construct dataloaders from NASGraphDataset

    Parameters
    ----------
    dataset : NASGraphDataset

    batch_size : int, optional
        by default 50
    train_test_split : float, optional
        number between 0 and 1, the proportion of the dataset to be used for training, by default 1
    subset : int, optional
        choose only <subset> many samples from the dataset. Set 0 for disabling, i.e. whole dataset. by default 0
    seed : int, optional
        set seed for reproduceability
    validation : bool, optional
        also output a validation set e.g. for hyperparam tuning

    Returns
    -------
    tuple(GraphDataLoader, (GraphDataLoader), GraphDataLoader)
        training dataloader to be used in CostPredictor.train() and test/validation dataloader if train_test_split > 0,
        else len(test_dataloader) == 0
    """
    np.random.seed(seed)
    valid_indices = np.arange(len(dataset), dtype=int)
    if subset:
        num_examples = subset
    else:
        num_examples = len(valid_indices)
    num_train = int(num_examples * train_test_split)
    if validation:
        num_val = int((num_examples - num_train) / 2)
    else:
        num_val = 0

    indices = np.random.choice(valid_indices, size=num_examples, replace=False)
    print("Indices", indices)
    train_indices = indices[:num_train]
    val_indices = indices[num_train:num_train + num_val]
    test_indices = indices[num_train + num_val:]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_dataloader = GraphDataLoader(
        dataset, sampler=train_sampler, batch_size=batch_size, drop_last=False
    )
    val_dataloader = GraphDataLoader(
        dataset, sampler=val_sampler, batch_size=batch_size, drop_last=False
    )
    test_dataloader = GraphDataLoader(
        dataset, sampler=test_sampler, batch_size=batch_size, drop_last=False
    )
    if validation:
        return train_dataloader, val_dataloader, test_dataloader
    else:
        return train_dataloader, test_dataloader


def get_input_feature_size(dataset, fea_name="features"):
    return dataset[0][0].ndata[fea_name].shape[1]
