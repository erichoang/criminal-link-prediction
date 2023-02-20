#!/usr/bin/env python3
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import ParameterGrid
from torch import randperm, ones, zeros, cat, device, cuda
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from torch.utils.tensorboard.writer import SummaryWriter
from torch_geometric.utils import from_networkx
from dgi import DGI
from dgi_net import DGINet
from process_burglary_network import load_graph_from_json_file, remove_node_without_emb_case_summary, reverse_array_from_string
from utils import TransductiveGraphSampler, InductiveGraphSampler


def train_model(model, optimizer, loss, patience, train_graph, validation_graph, metric):
    """
    Perform model training with early stopping
    :param model: pytroch model
    :param optimizer: model optimizer
    :param loss: criterion for the model
    :param patience: early stopping patience
    :param train_graph: graph to train on
    :param validation_graph: graph for validation
    :param metric: what metric to use
    """
    epoch_eval = 50
    patience_counter = 0
    while patience_counter < patience:
        for e in range(epoch_eval):
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

g = load_graph_from_json_file("dataset/israel_lea_inp_burglary/israel_lea_inp_burglary_v2_crime_id_network.json")
remove_node_without_emb_case_summary(g)
for node in g.nodes():
    temp_list = reverse_array_from_string(g.nodes[node]['emb_case_summary'])
    g.nodes[node]['emb_case_summary'] = temp_list

# graph_dataset = from_networkx(g, group_node_attrs=['emb_case_summary'], group_edge_attrs=['weight'])

# Evaluation settings
parameter_grid = {
    "hidden_dim": [128, 256, 512],
    "out_dim": [64, 32],
    "conv_hops": [2, 3, 4, 5],
    "gnn_arch": ["SGC", "GCN", "GIN", "GAT"],
    "normalization": ["batch", "instance", "graph", None],
    "dropout": [None, 0.3, 0.5],
    "lof_neighbors": [10, 20, 30],
    "leaf_size": [20, 30, 40],
    "metric": ["minkowski", "cosine"]
}
in_features = 64

step_size = 0.001
weight_decay = 1e-5

epoch_amount = 400
patience = 20
lowest_loss = 1e9

node_amount = graph_dataset.x.shape[0]
node_feature_amount = graph_dataset.x.shape[1]
dev = device("cuda") if cuda.is_available() else device("cpu")
# Model training

transductive_sampler = TransductiveGraphSampler(10, "crime_type")

inductive_sampler = InductiveGraphSampler(10, "crime_type")

for params in ParameterGrid(parameter_grid):

    model = DGI(
        in_features,
        params["hidden_dim"],
        params["out_dim"],
        params["conv_hops"],
        params["gnn_arch"],
        normalize=params["normalization"],
        dropout=params["dropout"],
        lof_neighbors=params["lof_neighbors"],
        lof_leaf_size=params["leaf_size"],
        lof_metric=params["metric"]
    ).to(dev)

    optimizer = Adam(model.parameters(), step_size, weight_decay=weight_decay)
    criterion = BCEWithLogitsLoss()

    model.train()

    model = train_model(model, optimizer, criterion, transductive_sampler)

for epoch in range(epoch_amount):
    optimizer.zero_grad()

    original_features = graph_dataset.x.to(dev)
    edge_index = graph_dataset.edge_index.to(dev)

    original_labels = ones(node_amount)
    corrupted_labels = zeros(node_amount)
    labels = cat([original_labels, corrupted_labels]).to(dev)

    logits = model(original_features, corrupted_features, edge_index)

    loss = criterion(logits, labels)

    #if loss < lowest_loss:
    #    patience -= 1


    print(f"Epoch # {epoch + 1}")
    print(f"Loss: {loss.item()}")

    if patience == 0:
        break

model.eval()
node_embeddings = graph_dataset.x.to(dev)
edge_index= graph_dataset.edge_index.to(dev)
node_embeddings = model.forward_embedding(node_embeddings, edge_index) \
                       .to(device("cpu")) \
                       .detach() \
                       .numpy()

print("Scoring outliers")
outlier_predictor = LocalOutlierFactor()
dimensionality_reducer = PCA(n_components=2)
scores = outlier_predictor.fit_predict(node_embeddings)
coordinates = dimensionality_reducer.fit_transform(node_embeddings)

for i, node in enumerate(g.nodes()):
    del g.nodes[node]['emb_case_summary']
    del g.nodes[node]["list_oid"]
    g.nodes[node]["outlier_score"] = scores[i]
    g.nodes[node]["embedding_x"] = coordinates[i, 0]
    g.nodes[node]["embedding_y"] = coordinates[i, 1]

write_gexf(g, "processed_graph.gexf")
