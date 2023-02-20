#!/usr/bin/env python3

from statistics import mean
from torch.autograd import grad
from torch.nn import Module, BCEWithLogitsLoss
from torch.nn.functional import softmax
from torch.optim import Adam, Optimizer
from torch import jit
from typing import Iterable, Tuple, List, Optional, Iterator
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_networkx
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score, f1_score
from dgi.dgi import DGI
from utils import TransductiveGraphSampler, InductiveGraphSampler
from datetime import datetime
import random
import time
import numpy as np
import networkx as nx
import torch


class Evaluation:
    """ Train and evaluate a model """


    def __init__(self,
                 model: Module,
                 optimizer: Optimizer,
                 criterion: Module,
                 params: dict,
                 graph: nx.Graph,
                 device: torch.device):
        """ Initializes model evaluation over given parameters """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion.to(device)
        self.params = params
        self.graph = graph
        self.device = device


    def train(self, batches: DataLoader) -> List[float]:
        """
        Train the model
        :param batches: dataloader that outputs batches
        :returns: list of batch losses
        """
        losses = []
        self.model.train()
        for batch in batches:
            self.optimizer.zero_grad()
            outputs = []
            labels = []
            for graph in batch:
                X = graph.x.to(self.device)
                edge_index = graph.edge_index.to(self.device)
                y = graph.y
                logits = self.model(X, edge_index)
                outputs.append(logits)
                labels.append(y)
            outputs = torch.stack(outputs)
            labels = torch.tensor(labels).to(self.device)
            loss = self.criterion(outputs, labels)
            losses.append(loss.item())
            loss.backward()
            self.optimizer.step()
        return losses


    @torch.no_grad()
    def test(self, test_data: Data) -> dict:
        """
        Performs test run over current model and given test data
        :param test_data: List of graphs for testing
        :returns: dictionary of evaluation metrics
        """
        self.model.eval()
        predictions = []
        labels = []
        for data in test_data:
            X = data.x.to(self.device)
            edge_index = data.edge_index.to(self.device)
            logits = self.model(X, edge_index)
            prediction = softmax(logits, dim=0).argmax().item()
            predictions.append(prediction)
            labels.append(data.y.item())
        return {
            "accuracy_score": accuracy_score(labels, predictions),
            "f1_score": f1_score(labels, predictions)
        }


    def make_data(self, splits: Tuple[nx.Graph, nx.Graph]) -> Tuple[DataLoader, Data]:
        """
        Create train dataloader and test data from splits
        :param splits: Tuple of splits from graph sampler
        :returns: Tuple of DataLoader and Data instances
        """
        train_dataset = from_networkx(splits[0], group_node_attrs=['emb_case_summary'], group_edge_attrs=['weight'])
        test_dataset = from_networkx(splits[1], group_node_attrs=['emb_case_summary'], group_edge_attrs=['weight'])
        train_loader = DataLoader(train_dataset, batch_size=self.params["batch_size"])
        return ()


    def set_splits(self):
        """
        Initialize splits for dataset training and evaluation
        """
        if self.params["validation_strategy"] == "inductive_downsample":
            self.splits = InductiveGraphSampler(
                self.params["fold_amount"],
                self.params["stratified_sample"],
                self.params["seed"],
                True).split(self.graph, self.params["downsample_class"], self.params["downsample_rate"])
        elif self.params["validation_strategy"] == "transductive_downsample":
            self.splits = TransductiveGraphSampler(
                self.params["fold_ampunt"],
                self.params["stratified_sample"],
                self.params["seed"],
                True).split(self.graph, self.params["downsample_class"], self.params["downsample_rate"])
        else:
            raise Exception("Validation strategy not legal")


    def evaluate_setting(self) -> Iterator[Tuple[dict, dict]]:
        """ Evaluate hyperparameter setting """
        self.set_splits()
        patience = self.params["patience"]
        epochs_left = self.params["epochs"]
        while patience > 0 and epochs_left > 0:
            for splits in self.splits:
                self.train(splits[0])
                test_result = self.test(splits[1])
                test_metrics.append(test_result)
                if len(splits) == 3:
                    valid_result = self.test(splits[2])
                    valid_metrics.append(valid_result)
            end_time = time.time()
            batch_loss = ";".join([str(ll) for l in batch_loss for ll in l])
            test_metrics = {
                "test_accuracy_score": mean([s["accuracy_score"] for s in test_metrics]),
                "test_f1_score": mean([s["f1_score"] for s in test_metrics])
            }
            report["start_timestamp"] = start_time
            report["end_timestamp"] = end_time
            report["batch_loss"] = batch_loss
            report.update(test_metrics)
            if valid_metrics:
                valid_metrics = {
                    "valid_accuracy_score": mean([s["accuracy_score"] for s in valid_metrics]),
                    "valid_f1_score": mean([s["f1_score"] for s in valid_metrics])
                }
            report.update(valid_metrics)
        yield report, self.model.state_dict()


class Experiment:
    """ Class for carrying out experiments """

    def __init__(self,
                 parameter_space: ParameterGrid,
                 graph: nx.Graph):
        """
        Initialize experiment
        :param parameter_space: Parameter space of model evaluation
        :param graph: Graph to be used for evaluation
        """
        self.parameter_space = parameter_space
        self.graph = graph
        self.criterion = BCEWithLogitsLoss()
        self.best_model: Optional[Module]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.set_random_seed()


    def set_random_seed(self):
        """ Sets random seed for the experiment """
        self.seed = self.parameter_space.param_grid[0]["seed"][0]
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)


    def get_experiment(self, params: dict) -> Tuple[Module, Optimizer]:
        """
        Initialize experiment classes
        :param params: parameter dictionary
        """
        model = DGI(
            in_len=params["in_len"],
            out_len=params["out_len"],
            hidden_len=params["hidden_len"],
            gin_hidden=params.get("gin_hidden", 64),
            gnn_arch=params["gnn_arch"],
            gnn_layers=params["gnn_layers"],
            gin_linear_amount=params.get("gin_linear_amount", 2),
            normalize=params["normalize"],
            dropout=params["dropout_ratio"],
            activation=params["activation"],
            readout=params["readout"],
            readout_point=params["readout_point"],
            lof_neighbors=params["lof_neighbors"],
            lof_leaf_size=params["lof_leaf_size"],
            lof_metric=params["lof_metric"]
        )
        optimizer = Adam(
            params=model.parameters(),
            lr=params["learning_rate"],
            weight_decay=params["weight_decay"]
        )
        return model, optimizer


    def evaluate(self) -> Iterator[Evaluation]:
        """
        Iterate over parameter space to evaluate different training settings
        """
        for params in self.parameter_space:
            model, optimizer = self.get_experiment(params)
            yield Evaluation(model, optimizer, self.criterion, params, self.graph, self.device)
