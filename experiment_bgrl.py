#!/usr/bin/env python3
from sklearn.model_selection import ParameterGrid
from torch_geometric.loader import DataLoader
from data_preparation import prepare_roxsd
from bgrl.experiment import Experiment
import torch

dataset = prepare_roxsd()

dataset.x = dataset.x.type(torch.FloatTensor)

settings = {
    "epochs": [600],
    "seed": [123],
    "batch_size": [5],
    "in_len": [dataset.x.shape[1]],
    "out_len": [64],
    "hidden_len": [64],
    "gnn_arch": ["gcn"],
    "gnn_layers": [3],
    "normalize": [None, "graph", "batch"],
    "dropout_ratio": [0, 0.3, 0.5],
    "activation": ["prelu"],
    "predictor_hidden_dim": [64],
    "predictor_linear_amount": [3],
    "learning_rate": [0.0001],
    "weight_decay": [0.01],
    "decay_warmup_steps": [50],
    "transform": ["edge_pertrubation"],
    "anomaly_detector": ["lof"],
    "lof_neighbors": [20],
    "lof_leaf_size": [30],
    "lof_metric": ["cosine"],
    "patience_metric": ["roc_auc"],
    "momentum": [0.1],
    "min_change": [0],
    "patience": [20]
}

grid = ParameterGrid(settings)

for s in grid:
    experiment = Experiment(s, dataset)
    loader = DataLoader([experiment.dataset], shuffle=True, batch_size=s["batch_size"])
    for result in experiment.train(loader, experiment.dataset):
        print(result)
