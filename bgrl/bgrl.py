""" BGRL implementation """
from copy import deepcopy
from torch.autograd.grad_mode import no_grad
from torch.nn import Module
from torch.nn.functional import cosine_similarity
from dgi.dgi import GCNLayer, GINLayer
from sklearn.base import BaseEstimator
import torch.nn as n
import torch
import numpy as np


class BGRL(Module):
    """ BGRL model class """


    def __init__(self, encoder: Module, predictor: Module, detector: BaseEstimator) -> None:
        super().__init__()

        self.online_encoder = encoder
        self.online_predictor = predictor

        self.target_encoder = deepcopy(encoder)
        self.target_encoder.reset_parameters()

        self.detector = detector

        for parameter in self.target_encoder.parameters():
            parameter.requires_grad = False


    def forward(self, X_online, edge_index_online, X_target, edge_index_target):
        """ Forward pass on BGRL model """

        H = self.online_encoder(X_online, edge_index_online)
        Z = self.online_predictor(H)

        H_target = self.target_embedding(X_target, edge_index_target)

        return Z, H_target


    def trainable_params(self):
        return list(self.online_encoder.parameters()) + list(self.online_predictor.parameters())


    @no_grad()
    def update_target(self, momentum):
        """ Update target encoder weights """
        for param_o, param_t in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_t.mul_(momentum).add_(param_o.data, alpha=1. - momentum)


    @no_grad()
    def target_embedding(self, X, edge_index):
        """ Compute target embedding based on the target encoder """
        return self.target_encoder(X, edge_index).detach()


    def predict(self, X, edge_index):
        """
        Make predictions
        :param X: Node attribute matrix
        :param edge_index: Edge list
        :returns: Anomaly labels
        """
        embeddings = self.target_embedding(X, edge_index).to(torch.device("cpu")).numpy()
        return self.detector.fit_predict(embeddings)


    def calc_loss(self, Z_1: torch.Tensor, H_1: torch.Tensor, Z_2: torch.Tensor, H_2: torch.Tensor) -> torch.Tensor:
        """
        Calculate symmetric loss for BGRL model
        :param Z_1: First prediction of target
        :param H_1: First target embedding
        :param Z_2: Second prediction of target
        :param H_2: Second target embedding
        :returns: Symmetric loss to be optimized
        """
        return 2 - cosine_similarity(Z_1, H_1, dim=-1).mean() - cosine_similarity(Z_2, H_2, dim=-1).mean()


class Predictor(Module):
    """ Predictor neural network """


    def __init__(self, in_dim, out_dim, hidden_dim, linear_amount) -> None:
        super().__init__()
        self.layers = []
        for i in range(linear_amount):
            in_len = None
            out_len = None
            if i == 0:
                in_len = in_dim
            else:
                self.layers.append(n.PReLU())
                in_len = hidden_dim
            if i + 1 == linear_amount:
                out_len = out_dim
            else:
                out_len = hidden_dim
            self.layers.append(n.Linear(in_len, out_len))
        self.layers = n.Sequential(*self.layers)


    def forward(self, x):
        return self.layers(x)


class GraphNet(Module):
    """ Graph neural network """


    def __init__(self,
                 in_len: int,
                 out_len: int,
                 hidden_len: int = 64,
                 gin_hidden: int = 64,
                 gnn_arch: str = "gcn",
                 gnn_layers: int = 5,
                 gin_linear_amount: int = 2,
                 normalize: str = None,
                 dropout: float = None,
                 activation: str = "relu") -> None:
        super().__init__()
        self.gnn = n.ModuleList()
        for i in range(gnn_layers):
            if i == 0:
                layer = self.get_graph_layer(
                    in_len,
                    hidden_len,
                    gin_hidden,
                    gnn_arch,
                    gnn_layers,
                    gin_linear_amount,
                    normalize,
                    dropout,
                    activation
                )
            else:
                layer = self.get_graph_layer(
                    hidden_len,
                    hidden_len,
                    gin_hidden,
                    gnn_arch,
                    gnn_layers,
                    gin_linear_amount,
                    normalize,
                    dropout,
                    activation
                )
            self.gnn.append(layer)


    def reset_parameters(self):
        """ Reset model parameters """
        for layer in self.gnn:
            layer.init_params()


    def get_graph_layer(self, in_len, out_len, gin_hidden, gnn_arch, gnn_layers, linear_amount, normalize, dropout, activation):
        """
        Select GNN layer by architecture and initialize it
        """
        if gnn_arch == "gin":
            layer = GINLayer(in_len, out_len, linear_amount, gin_hidden, normalize=normalize, dropout=dropout, activation=activation)
        elif gnn_arch == "gcn":
            layer = GCNLayer(in_len, out_len, normalize, dropout, activation)
        else:
            raise Exception("Not a valid GNN architecture")
        return layer

    def forward(self, X, edge_index):
        H = X
        for layer in self.gnn:
            H = layer(H, edge_index)
        return H


class CosineDecayScheduler:
    def __init__(self, max_val, warmup_steps, total_steps):
        self.max_val = max_val # learning rate
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def get(self, step):
        if step < self.warmup_steps:
            return self.max_val * step / self.warmup_steps
        elif self.warmup_steps <= step <= self.total_steps:
            return self.max_val * (1 + np.cos((step - self.warmup_steps) * np.pi /
                                              (self.total_steps - self.warmup_steps))) / 2
        else:
            raise Exception("Step too high")
