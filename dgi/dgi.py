#!/usr/bin/env python3
from sklearn.neighbors import LocalOutlierFactor
from torch.nn.modules.activation import PReLU, ReLU
from torch_geometric.nn import SGConv, BatchNorm, InstanceNorm, GraphNorm, GATConv, GINConv, GCNConv, LayerNorm
from torch.nn.init import xavier_normal_ as normal, zeros_ as zeros
import torch.nn as n
import torch.nn.functional as fn
import torch

class GNNLayer(n.Module):
    """ Base class for GNN layers """


    def get_normalization_layer(self, arch: str, in_len: int) -> n.Module:
        """
        Select normalization layer
        :param arch: normalization architecture
        :returns: Initialized normalization layer
        """
        layer = None
        if arch == "batch":
            layer = BatchNorm(in_len)
        elif arch == "instance":
            layer = InstanceNorm(in_len, affine=True)
        elif arch == "layer":
            layer = LayerNorm(in_len)
        elif arch == "graph":
            layer = GraphNorm(in_len)
        else:
            raise Exception("No such normalization")
        return layer


    def get_activation_layer(self, arch: str) -> n.Module:
        """
        Select activation layer
        :param arch: activation architecture
        :returns: Initialized activation layer
        """
        layer = None
        if arch == "relu":
            layer =  n.ReLU()
        elif arch == "prelu":
            layer = n.PReLU()
        else:
            raise Exception("No such activation")
        return layer


    def init_params(self):
        """
        Initialize parameters for GNN layer
        """
        raise NotImplementedError("Parameter initialization is not implemeted in the base class")


class GCNLayer(GNNLayer):
    """ Single layer of GCN network """


    def __init__(self,
                in_len: int,
                out_len: int,
                normalize: str = None,
                dropout: float = None,
                activation: str = "relu"):
        """
        Initialize GCN network layer
        :param in_len: in_channels of GCNConv
        :param out_len: out_channels of GCNConv
        :param normalize: type of normalization of the layer
        :param dropout: dropout rate, if None no dropout will be used
        :param activation: activation type for the layer
        """
        super(GCNLayer, self).__init__()
        self.layer = GCNConv(in_len, out_len)
        self.normalization = self.get_normalization_layer(normalize, out_len) if normalize else None
        self.activation = self.get_activation_layer(activation)
        self.dropout = n.Dropout(dropout) if dropout else None
        self.init_params()


    def forward(self, X: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Does a single forward pass thorugh GCN layer
        :param X: Node feature matrix
        :param edge_index: adjacency list of the graph
        :returns: updated node embeddings
        """
        H = self.layer(X, edge_index)
        if self.normalization:
            H = self.normalization(H)
        H = self.activation(H)
        if self.dropout:
            H = self.dropout(H)
        return H


    def init_params(self):
        """
        Initialize GCN parameters
        """
        normal(self.layer.lin.weight)
        zeros(self.layer.bias)


class GINLayer(GNNLayer):
    """ Single layer of GIN network """


    def __init__(self,
                in_len: int,
                out_len: int,
                linear_num: int = 1,
                hidden_dim: int = 64,
                gnn_eps: float = 0,
                train_eps: bool = False,
                normalize: str = None,
                dropout: float = None,
                activation: str = "relu"):
        """
        Initialize GIN network layer
        :param in_len: Node feature dimentions
        :param out_len: Output dimentaions of neutal network
        :param linear_num: Number of liner layers of the neural network, must be at least 1
        :param hidden_dim: Hidden dimentions of hidden layer of neural network
        :param gnn_eps: epsilon value
        :param train_eps: should epsilon alue be trainable
        :param normalize: type of normalization of the layer
        :param dropout: dropout rate, if None no dropout will be used
        :param activation: activation type for the layer
        """
        super(GINLayer, self).__init__()
        self.linear_amount: int = linear_num
        self.linear_input = n.Linear(in_len, hidden_dim if linear_num > 1 else out_len)
        self.gnn = GINConv(self.linear_input, gnn_eps, train_eps)
        self.linear_layers = n.ModuleList()
        self.init_mlp(out_len, hidden_dim, activation, dropout, normalize)
        self.init_params()


    def init_mlp(self, out_len, hidden_dim, act_arch, dropout, norm_arch=None):
        """
        Initializes GIN neural network
        :param out_len: output dimention
        :param hidden_dim: hidden dimentions of neural network
        :param norm_arch: normalization layer name
        :param act_arch: activation layer name
        """
        for i in range(self.linear_amount):
            if i > 0:
                layer = None
                if i + 1 < self.linear_amount:
                    layer = n.Linear(hidden_dim, hidden_dim)
                else:
                    layer = n.Linear(hidden_dim, out_len)
                self.linear_layers.append(layer)
            if norm_arch:
                normalization = self.get_normalization_layer(
                    norm_arch,
                    hidden_dim if i + 1 < self.linear_amount else out_len)
                self.linear_layers.append(normalization)
            activation = self.get_activation_layer(act_arch)
            self.linear_layers.append(activation)
            if dropout:
                self.linear_layers.append(n.Dropout(dropout))


    def init_params(self):
        """Initialize gnn parameters"""
        normal(self.linear_input.weight)
        zeros(self.linear_input.bias)
        for l in self.linear_layers:
            if isinstance(l, n.Linear):
                normal(l.weight)
                zeros(l.bias)


    def forward(self, X: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Does a forward pass through GIN layer
        :param X: Node feature matrix
        :param edge_index: adjacency list of network
        :returns: GIN forward pass result
        """
        H = self.gnn(X, edge_index)
        for layer in self.linear_layers:
            H = layer(H)
        return H


class DGI(n.Module):
    """ Main Deep Graph infomax implementation """

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
                 activation: str = "relu",
                 readout: str = "mean",
                 readout_point: str = "layer",
                 lof_neighbors: int = 20,
                 lof_leaf_size: int = 30,
                 lof_metric: str = "minkowski"):
        """
        GNN config
        :param in_len: node feature amount
        :param out_len: output embedding dimension
        :param hidden_len: hidden representation length
        :param gin_hidden: hidden dimensions of GIN neural network
        :param gnn_arch: type of graph neural network
        :param gnn_layers: amount of GNN layers
        :param gin_linear_amount: amount of linear layers in GIN layer
        :param normalize: type of normalization used
        :param dropout: dropout rate
        :param activation: activation function
        :param readout: Graph readout function type, can be "mean, "max" or "sum"
        :param readout_point: if "layer" was passed - applies readout after each GNN layer and concatenates result, if "output" - applies readout after last GNN layer
        :param lof_neighbors: LOF neighbor amount
        :param lof_leaf_size: LOF tree leaf size
        :param lof_metric: LOF distance metric
        """
        super(DGI, self).__init__()
        self.gnn = n.ModuleList()
        # setup message passing architecture
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
        self.readout_type = readout
        self.readout_point = readout_point
        self.out = n.Linear(hidden_len, out_len)
        if readout_point == "layer":
            self.discriminator_similarity = n.Bilinear(out_len, out_len + (hidden_len * (gnn_layers - 1)), 1)
        elif readout_point == "output":
            self.discriminator_similarity = n.Bilinear(out_len, out_len, 1)
        else:
            raise Exception("No such readout point")
        # discriminator config
        self.discriminator_activation = n.LogSigmoid()
        # outlier detector config
        self.detector = LocalOutlierFactor(
            n_neighbors=lof_neighbors,
            leaf_size=lof_leaf_size,
            metric=lof_metric,
            n_jobs=-1
        )
        # init linear layer weights
        self.init_weight_bias(self.out)
        self.init_weight_bias(self.discriminator_similarity)

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

    def init_weight_bias(self, layer):
        """
        Initializes layer weight and bias parameters
        :param layer: Neural network layer
        """
        normal(layer.weight)
        zeros(layer.bias)

    def forward(self, X, X_corrupted, edge_index):
        """
        Fowrward pass of the network
        :param X: matrix of node features
        :param X_corrupted: corrupted features
        :param edge_index: list of network edges
        :returns: scores of vectors relative to summary vector (original first, then corrupted)
        """
        embeddings, summary = self.forward_embedding(X, edge_index)
        corrupted_embeddings, corrupted_summary = self.forward_embedding(X_corrupted, edge_index)
        original_scores = self.discriminator(embeddings, summary)
        corrupted_scores = self.discriminator(corrupted_embeddings, corrupted_summary)
        return torch.cat([original_scores, corrupted_scores])

    def readout(self, H, dim: int = 0):
        """
        Apply readout to the hidden representation
        :param H: matrix of node hidden representations
        :param dim: dimension to collapse
        :returns: aggregate of all hidden representations
        """
        if self.readout_type == "mean":
            return torch.mean(H, dim)
        elif self.readout_type == "max":
            return torch.max(H, dim).values
        elif self.readout_type == "sum":
            return torch.sum(H, dim)
        else:
            raise Exception("No such readout function")

    def gnn_layer_agg(self, X, edge_index):
        """
        Does a forward pass through GNN and concatenates each layer readout
        :param X: Node features
        :param edge_index: adjacency list
        :returns: tuple of hidden representations and graph readout
        """
        H = X
        readouts = []
        for layer in self.gnn:
            H = layer(H, edge_index)
            if layer is self.gnn[-1]:
                H = self.out(H)
            readouts.append(self.readout(H))
        return H, torch.cat(readouts)

    def gnn_output_agg(self, X, edge_index):
        """
        Does a forward pass through GNN and aggregates last output readout
        :param X: Node features
        :param edge_index: adjacency list
        :returns: tuple of hidden representations and graph readout
        """
        H = X
        for layer in self.gnn:
            H = layer(H, edge_index)
        H = self.out(H)
        return H, self.readout(H)

    def discriminator(self, H, s):
        """
        Discriminator function that returns probability of node being in the graph
        :param H: embedding matrix
        :oaram s: summary vecotr
        """
        ex_s = s.expand(H.shape[0], -1)
        results = self.discriminator_activation(self.discriminator_similarity(H, ex_s))
        return torch.squeeze(results)

    def forward_embedding(self, X, edge_index):
        """
        Train embeddings
        :param X: node features
        :param edge_index: adjacency list
        :returns: embeddings passed through GNN architecture and readout Tuple
        """
        if self.readout_point == "layer":
            H, readout = self.gnn_layer_agg(X, edge_index)
        elif self.readout_point == "output":
            H, readout = self.gnn_output_agg(X, edge_index)
        else:
            raise Exception("No such readout strategy")
        return H, readout

    def corrupt_network(self, X):
        """
        Corrupt original network nodes by shuffling node features
        :param X: Node feature matrix
        :retruns: Torch matrix of shuffled features
        """
        shuffle_perm = torch.randperm(X.shape[1])
        return X[:, shuffle_perm]

    def predict(self, X, edge_index):
        """
        Predict outliers using LOF
        :param X: node features
        :param edge_index: adjacency list
        :returns: outlier scores (1 for inlier, -1 for outlier) as numpy vector
        """
        embeddings = self.forward_embedding(X, edge_index)[0] \
            .to(torch.device("cpu")) \
            .detach() \
            .numpy()
        return self.detector.fit_predict(embeddings)
