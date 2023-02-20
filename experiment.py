#!/usr/bin/env python3
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from sklearn.model_selection import ParameterGrid
from utils import roxsd_to_dataset, seed_everything
from dgi.dgi import DGI
from sys import maxsize
import pickle
import torch
import networkx as nx


def init_learning(settings: dict):
    """
    Initializes model and optimizer for learning
    """
    model = DGI(
        in_len=settings["in_len"],
        out_len=settings["out_len"],
        hidden_len=settings["hidden_len"],
        gin_hidden=settings["gin_hidden"],
        gnn_arch=settings["gnn_arch"],
        gnn_layers=settings["gnn_layers"],
        normalize=settings["normalize"],
        dropout=settings["dropout_ratio"],
        activation=settings["activation"],
        readout=settings["readout"],
        readout_point=settings["readout_point"],
        lof_neighbors=settings["lof_neighbors"],
        lof_leaf_size=settings["lof_leaf_size"],
        lof_metric=settings["lof_metric"]
    )
    optimizer = Adam(
        params=model.parameters(),
        lr=settings["learning_rate"],
        weight_decay=settings["weight_decay"]
    )
    model = model.to(device)
    model.train()
    return model, optimizer


with open("./dataset/roxsd_v2_core_graphs/graph_roxanne_roxsd_v2.pkl", "rb") as f:
    graph = pickle.load(f)

dataset = roxsd_to_dataset(graph)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

settings = {
    "epoch_amount": [600],
    "seed": [123],
    "in_len": [dataset.x.shape[1]],
    "out_len": [10, 20, 30],
    "hidden_len": [20, 30, 40],
    "gin_hidden": [20, 30, 40],
    "gnn_arch": ["gcn", "gin"],
    "gnn_layers": [2, 3, 4],
    "normalize": [None, "graph", "batch"],
    "dropout_ratio": [0, 0.3, 0.5],
    "activation": ["relu", "prelu"],
    "readout": ["mean", "max", "sum"],
    "readout_point": ["layer", "output"],
    "lof_neighbors": [5, 10, 20],
    "lof_leaf_size": [20, 30, 40],
    "lof_metric": ["minkowski", "cosine"],
    "learning_rate": [0.001, 0.0001],
    "weight_decay": [0, 0.05, 0.01],
    "min_change": [0],
    "patience": [20]
}


X = dataset.x.to(torch.float32).to(device)
edge_index = dataset.edge_index.to(device)

for i, params in enumerate(grid):
    model, optimizer = init_learning(params)
    patience_counter = 0
    last_loss = torch.tensor(maxsize).to(device)
    for epoch in range(1, params["epoch_amount"] + 1):
        print(f"Epoch: {epoch}")
        corrupted = model.corrupt_network(X)
        original_labels, corrupted_labels = torch.full((dataset.x.shape[0],), 1), torch.full((dataset.x.shape[0],), 0)
        labels = torch.cat([original_labels, corrupted_labels], dim=0).to(torch.float32).to(device)
        optimizer.zero_grad()
        results = model(X, corrupted, edge_index)
        loss = criterion(results, labels)
        print(f"Loss: {loss}")
        loss.backward()
        optimizer.step()
        if (last_loss - loss) < params["min_change"]:
            patience_counter += 1
        if patience_counter >= params["patience"]:
            print("Early stopping!")
            break
        last_loss = loss
    model.eval()
    outlier_predictions = model.predict(X, edge_index)
    print("Writing predictions to disc")
    torch.save(model, f"./results/model_settings_{i}.pt")
    outliers_predicted = 0
    for node_name, prediction in zip(graph.nodes, outlier_predictions):
        graph.nodes[node_name]["outlier_prediction"] = prediction
        if "None" in node_name and prediction == -1:
            outliers_predicted += 1
    nx.write_gexf(graph, f"./results/model_settings_{i}.gexf")
    with open(f"./results/model_settings_{i}.txt", "w") as f:
        f.write(str(params))
        f.write("\n")
        f.write(str(outliers_predicted))
