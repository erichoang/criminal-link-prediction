#!/usr/bin/env python3
from torch.optim import AdamW
from bgrl.bgrl import BGRL, Predictor, CosineDecayScheduler, GraphNet
from dig.sslgraph.method.contrastive.views_fn import EdgePerturbation, NodeAttrMask, UniformSample, RWSample
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, recall_score, precision_score
from typing import Dict
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import random
import networkx as nx
import numpy as np
import torch

class Experiment:
    """ Class for carrying out experiments """

    def __init__(self,
                 params: Dict,
                 dataset: Data):
        """
        Initialize experiment
        :param parameter_space: Parameter space of model evaluation
        :param dataset: Data objec
        """
        self.parameters = params
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.set_random_seed()
        self.encoder = GraphNet(
            in_len=params["in_len"],
            out_len=params["out_len"],
            hidden_len=params["hidden_len"],
            gin_hidden=params.get("gin_hidden", 64),
            gnn_arch=params["gnn_arch"],
            gnn_layers=params["gnn_layers"],
            gin_linear_amount=params.get("gin_linear_amount", 2),
            normalize=params["normalize"],
            dropout=params["dropout_ratio"],
            activation=params["activation"]
        )
        self.predictor = Predictor(
            in_dim=params["out_len"],
            out_dim=params["out_len"],
            hidden_dim=params["predictor_hidden_dim"],
            linear_amount=params["predictor_linear_amount"]
        )
        self.detector = self.prepare_detector(params["anomaly_detector"])
        self.model = BGRL(self.encoder, self.predictor, self.detector)
        self.optimizer = AdamW(
            params=self.model.trainable_params(),
            lr=0.,
            weight_decay=params["weight_decay"]
        )
        self.opt_scheduler = CosineDecayScheduler(
            max_val=params["learning_rate"],
            warmup_steps=params["decay_warmup_steps"],
            total_steps=params["epochs"]
        )
        self.target_scheduler = CosineDecayScheduler(
            max_val=1 - params["momentum"],
            warmup_steps=0,
            total_steps=params["epochs"]
        )
        self.transform = self.prepare_transform(params["transform"])
        self.dataset = dataset
        self.patience_metric = params["patience_metric"]

    def set_random_seed(self):
        """ Sets random seed for the experiment """
        self.seed = self.parameters["seed"]
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)


    def prepare_transform(self, transform_type: str):
        """ Return transformation view based on setting """
        if transform_type == "attr_mask":
            return NodeAttrMask()
        if transform_type == "edge_pertrubation":
            return EdgePerturbation(add=True, drop=True)
        if transform_type == "random":
            return UniformSample()
        if transform_type == "walk_random":
            return RWSample()
        raise Exception(f"No such transformation: {transform_type}")


    def prepare_detector(self, detector_type: str):
        """ Return initialized detector """
        if detector_type == "lof":
            return LocalOutlierFactor(
                n_neighbors=self.parameters["lof_neighbors"],
                leaf_size=self.parameters["lof_leaf_size"],
                metric=self.parameters["lof_metric"],
                n_jobs=-1
            )
        if detector_type == "svm":
            return OneClassSVM(
                kernel=self.parameters["svm_kernel"],
                degree=self.parameters["svm_degree"],
                gamma=self.parameters["svm_gamma"],
                n_jobs=-1
            )
        raise Exception(f"No such anomaly detection algorithm: {detector_type}")


    def update_schedulers(self, epoch: int):
        """
        Learning rate update based on cosine decay
        :param epoch: Current learning epoch
        :returns: Target encoder momentum
        """
        lr = self.opt_scheduler.get(epoch)
        for g in self.optimizer.param_groups:
            g["lr"] = lr
        return 1 - self.target_scheduler.get(epoch)


    def train(self, data_loader: DataLoader, data: Data):
        """
        Iterate over parameter space to evaluate different training settings
        :param data_loader: DataLoader class that feeds training data to the model
        :param data: Evaluation data
        """
        self.model = self.model.to(self.device)
        epoch = 1
        patience = self.parameters["patience"]
        last_result = 999999999
        while epoch <= self.parameters["epochs"] and patience > 0:
            self.model.train()
            for batch in data_loader:
                self.optimizer.zero_grad()
                momentum = self.update_schedulers(epoch)
                batch = batch.to(torch.device("cpu"))
                transform_1, transform_2 = self.transform.views_fn(batch), self.transform.views_fn(batch)
                transform_1 = transform_1.to(self.device)
                transform_2 = transform_2.to(self.device)
                Z_1, H_target_1 = self.model(transform_1.x, transform_1.edge_index, transform_2.x, transform_2.edge_index)
                Z_2, H_target_2 = self.model(transform_2.x, transform_2.edge_index, transform_1.x, transform_1.edge_index)
                loss = self.model.calc_loss(Z_1, H_target_1, Z_2, H_target_2)
                loss.backward()
                self.optimizer.step()
                self.model.update_target(momentum)
            results = self.evaluate(data)
            yield results
            if last_result - results[self.patience_metric] <= self.parameters["min_change"]:
                patience -= 1
            last_result = results[self.patience_metric]
            epoch += 1


    def evaluate(self, data: Data):
        """
        Evaluate anomaly detection algorithm
        :param data: Data class with evaluation graph
        """
        self.model.eval()
        data = data.to(self.device)
        results = self.model.predict(data.x, data.edge_index)
        labels = data.y.to(torch.device("cpu")).numpy()
        return {
            "f1": f1_score(labels, results),
            "roc_auc": roc_auc_score(labels, results),
            "accuracy": accuracy_score(labels, results),
            "recall": recall_score(labels, results),
            "precision": precision_score(labels, results)
        }
