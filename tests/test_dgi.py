#!/usr/bin/env python3
from typing import Sequence
from torch.nn import ReLU, PReLU, Linear, Sequential
from torch.nn.modules import normalization
from torch_geometric.nn import BatchNorm, LayerNorm, InstanceNorm, GraphNorm, GCNConv, GINConv
from dgi.dgi import DGI, GNNLayer, GCNLayer, GINLayer
import pytest
import torch


@pytest.fixture
def dummy_graph():
    node_amount = 30
    feature_amount = 10
    edge_amount = 20
    X = torch.randn((node_amount, feature_amount))
    node_indexes = torch.arange(end=node_amount)
    sources = node_indexes.index_select(0, torch.randint(high=node_amount, size=(edge_amount,)))
    destinations = node_indexes.index_select(0, torch.randint(high=node_amount, size=(edge_amount,)))
    edge_index = torch.cat([sources, destinations]).reshape((2, edge_amount))
    return (X, edge_index)


def test_GNNLayer_get_notrmalization_layer_should_return_BatchNorm_if_arch_is_batch():
    layer = GNNLayer().get_normalization_layer("batch", 20)
    assert isinstance(layer, BatchNorm)


def test_GNNLayer_get_normalization_layer_should_return_InstanceNorm_if_arch_is_instance():
    layer = GNNLayer().get_normalization_layer("instance", 20)
    assert isinstance(layer, InstanceNorm)


def test_GNNLayer_get_normalization_layer_should_return_LayerNorm_if_arch_is_layer():
    layer = GNNLayer().get_normalization_layer("layer", 20)
    assert isinstance(layer, LayerNorm)


def test_GNNLayer_get_normalization_layer_should_return_GraphNorm_if_arch_is_graph():
    layer = GNNLayer().get_normalization_layer("graph", 20)
    assert isinstance(layer, GraphNorm)


def test_GNNLayer_get_activation_layer_should_return_ReLU_if_arch_is_relu():
    layer = GNNLayer().get_activation_layer("relu")
    assert isinstance(layer, ReLU)


def test_GNNLayer_get_activation_layer_should_return_PReLU_if_arch_is_prelu():
    layer = GNNLayer().get_activation_layer("prelu")
    assert isinstance(layer, PReLU)


def test_GCNLayer_forward_should_apply_GCN_layer_and_activation_if_no_normalization_nor_dropout_set(dummy_graph):
    X, edge_index = dummy_graph
    out_features = 10
    gcn_layer = GCNConv(X.shape[1], out_features)
    activation = ReLU()
    gcn_out = activation(gcn_layer(X, edge_index))

    gcn_custom = GCNLayer(X.shape[1], out_features)
    gcn_custom.layer.lin.weight = gcn_layer.lin.weight
    gcn_custom.layer.bias = gcn_layer.bias
    custom_out = gcn_custom(X, edge_index)
    assert torch.equal(gcn_out, custom_out)


def test_GCNLayer_forward_should_apply_dropout_if_dropout_passed_during_initialization(dummy_graph):
    X, edge_index = dummy_graph
    out_features = 10
    dropout_rate = 0.5
    gcn_custom = GCNLayer(X.shape[1], out_features, dropout=dropout_rate)
    gcn_custom.train()
    gcn_out = gcn_custom(X, edge_index)
    assert X.shape[0] * X.shape[1] * dropout_rate <= (gcn_out == 0).sum()


def test_GCNLayer_forward_should_apply_normalization_if_normalize_passed_during_initialization(dummy_graph):
    X, edge_index = dummy_graph
    out_features = 10
    norm_name = "batch"
    gcn_layer = GCNConv(X.shape[1], out_features)
    activation = ReLU()
    gcn_custom = GCNLayer(X.shape[1], out_features, normalize=norm_name)
    normalization = gcn_custom.normalization
    gcn_out = activation(normalization(gcn_layer(X, edge_index)))

    gcn_custom.layer.lin.weight = gcn_layer.lin.weight
    gcn_custom.layer.bias = gcn_layer.bias

    custom_out = gcn_custom(X, edge_index)
    assert torch.equal(gcn_out, custom_out)


def test_GINLayer_init_mlp_should_set_activation_and_normalization_layer_if_specified_during_init():
    in_len = 20
    out_len = 10
    linear_num = 1
    normalization = "batch"
    activation = "relu"
    gin_layer = GINLayer(in_len, out_len, linear_num=linear_num, normalize=normalization, activation=activation)
    assert len([l for l in gin_layer.linear_layers if isinstance(l, BatchNorm)]) == linear_num
    assert len([l for l in gin_layer.linear_layers if isinstance(l, ReLU)]) == linear_num


def test_GINLayer_init_mlp_should_set_additional_linear_layer_normalization_if_normalize_given():
    in_len = 20
    out_len = 10
    hidden_dim = 64
    linear_amount = 4
    normalize = "batch"
    gin_layer = GINLayer(in_len, out_len, linear_num=linear_amount, hidden_dim=hidden_dim, normalize=normalize)
    assert len([l for l in gin_layer.linear_layers if isinstance(l, BatchNorm)]) == linear_amount
    assert len([l for l in gin_layer.linear_layers if isinstance(l, ReLU)]) == linear_amount
    assert len([l for l in gin_layer.linear_layers if isinstance(l, Linear)]) == linear_amount - 1


def test_GINLayer_init_mlp_should_set_additional_linear_layers():
    in_len = 20
    out_len = 10
    hidden_dim = 64
    linear_amount = 4
    gin_layer = GINLayer(in_len, out_len, linear_num=linear_amount, hidden_dim=hidden_dim)
    linear_layers = [l for l in gin_layer.linear_layers if isinstance(l, Linear)]
    for l in linear_layers:
        assert l.in_features == hidden_dim
        if l is linear_layers[-1]:
            assert l.out_features == out_len
        else:
            assert l.out_features == hidden_dim


def test_GINLayer_forward_should_pass_gin_layer_with_activation(dummy_graph):
    X, edge_index = dummy_graph
    out_features = 10
    gin_layer = GINLayer(X.shape[1], out_features)
    gin = GINConv(gin_layer.linear_input)
    activation = ReLU()
    layer_out = gin_layer(X, edge_index)
    test_out = activation(gin(X, edge_index))
    assert torch.equal(layer_out, test_out)


def test_GINLayer_forward_should_pass_gin_layer_with_additional_linear_layers(dummy_graph):
    X, edge_index = dummy_graph
    out_features = 10
    linear_num = 3
    hidden_dim = 32
    gin_layer = GINLayer(X.shape[1], out_features, linear_num=linear_num, hidden_dim=hidden_dim)
    test_layers = [l for l in gin_layer.linear_layers]
    test_layers.insert(0, gin_layer.gnn.nn)
    test_network = Sequential(*test_layers)
    test_gin = GINConv(test_network)
    layer_out = gin_layer(X, edge_index)
    test_out = test_gin(X, edge_index)
    assert torch.equal(layer_out, test_out)


def test_GINLayer_forward_should_pass_gin_layer_with_normalization_if_normalization_specified(dummy_graph):
    X, edge_index = dummy_graph
    out_features = 10
    linear_num = 3
    hidden_dim = 32
    normalization = "batch"
    gin_layer = GINLayer(X.shape[1], out_features, linear_num=linear_num, hidden_dim=hidden_dim, normalize=normalization)
    test_layers = [l for l in gin_layer.linear_layers]
    test_layers.insert(0, gin_layer.gnn.nn)
    print(test_layers)
    test_network = Sequential(*test_layers)
    test_gin = GINConv(test_network)
    layer_out = gin_layer(X, edge_index)
    test_out = test_gin(X, edge_index)
    assert torch.equal(layer_out, test_out)


def test_GINLayer_forward_should_apply_dropout_if_dropout_rate_is_set(dummy_graph):
    X, edge_index = dummy_graph
    out_features = 10
    dropout_rate = 0.5
    gin_layer = GINLayer(X.shape[1], out_features, dropout=0.5)
    gin_layer.train()
    gin_out = gin_layer(X, edge_index)
    assert X.shape[0] * X.shape[1] * dropout_rate <= (gin_out == 0).sum()


def test_DGI_init_should_initialize_gcn_if_gnn_arch_is_gcn():
    in_len = 20
    logit_len = 1
    gnn_arch = "gcn"
    layer_amount = 3
    model = DGI(in_len, logit_len, gnn_layers=layer_amount, gnn_arch=gnn_arch)
    for layer in model.gnn:
        assert isinstance(layer, GCNLayer)
    assert len(model.gnn) == layer_amount


def test_DGI_init_should_initialize_gin_if_gnn_arch_is_gin():
    in_len = 20
    logit_len = 1
    gnn_arch = "gin"
    layer_amount = 3
    model = DGI(in_len, logit_len, gnn_layers=layer_amount, gnn_arch=gnn_arch)
    for layer in model.gnn:
        assert isinstance(layer, GINLayer)
    assert len(model.gnn) == layer_amount


def test_DGI_init_should_initialize_out_input_as_hidden_dim():
    in_len = 20
    logit_len = 1
    hidden_len = 64
    readout_point = "output"
    model = DGI(in_len, logit_len, hidden_len=hidden_len, readout_point=readout_point)
    assert model.out.in_features == hidden_len


def test_DGI_readout_should_average_if_readout_type_is_mean(dummy_graph):
    X, edge_index = dummy_graph
    in_len = 20
    logit_len = 1
    readout = "mean"
    dim = 0
    model = DGI(in_len, logit_len, readout=readout)
    model_out = model.readout(X, dim)
    assert torch.equal(torch.mean(X, dim), model_out)


def test_DGI_readout_should_get_max_if_readout_type_is_max(dummy_graph):
    X, edge_index = dummy_graph
    in_len = 20
    logit_len = 1
    readout = "max"
    dim = 0
    model = DGI(in_len, logit_len, readout=readout)
    model_out = model.readout(X, dim)
    assert torch.equal(torch.max(X, dim).values, model_out)


def test_DGI_readout_should_get_sum_if_readout_type_is_sum(dummy_graph):
    X, edge_index = dummy_graph
    in_len = 20
    logit_len = 1
    readout = "sum"
    dim = 0
    model = DGI(in_len, logit_len, readout=readout)
    model_out = model.readout(X, dim)
    assert torch.equal(torch.sum(X, dim), model_out)

def test_DGI_gnn_layer_agg_should_return_embedding_and_summary_concatenated_from_every_layer(dummy_graph):
    X, edge_index = dummy_graph
    in_len = X.shape[1]
    logit_len = 1
    gnn_layers = 3
    readout_point = "layer"
    model = DGI(in_len, logit_len, gnn_layers=gnn_layers, readout_point=readout_point)
    model_out = model.gnn_layer_agg(X, edge_index)
    H = X
    readouts = []
    for layer in model.gnn:
        H = layer(H, edge_index)
        if layer is model.gnn[-1]:
            H = model.out(H)
        readouts.append(model.readout(H))
    assert torch.equal(H, model_out[0])
    assert torch.equal(torch.cat(readouts), model_out[1])


def test_DGI_gnn_output_agg_should_return_embedding_and_summary_concatenated_from_last_layer(dummy_graph):
    X, edge_index = dummy_graph
    in_len = X.shape[1]
    logit_len = 1
    gnn_layers = 3
    readout_point = "layer"
    model = DGI(in_len, logit_len, gnn_layers=gnn_layers, readout_point=readout_point)
    model_out = model.gnn_output_agg(X, edge_index)
    H = X
    for layer in model.gnn:
        H = layer(H, edge_index)
    H = model.out(H)
    readout = model.readout(H)
    assert torch.equal(H, model_out[0])
    assert torch.equal(readout, model_out[1])


def test_DGI_readout_should_take_mean_of_embedding_if_readout_type_is_mean(dummy_graph):
    X, edge_index = dummy_graph
    in_len = X.shape[1]
    logit_len = 1
    gnn_layers = 3
    readout_point = "layer"
    readout = "mean"
    dim = 0
    model = DGI(in_len, logit_len, gnn_layers=gnn_layers, readout_point=readout_point, readout=readout)
    output = model.readout(X, dim=dim)
    assert torch.equal(output, torch.mean(X, dim=dim))


def test_DGI_readout_should_take_max_of_embedding_if_readout_type_is_max(dummy_graph):
    X, edge_index = dummy_graph
    in_len = X.shape[1]
    logit_len = 1
    gnn_layers = 3
    readout_point = "layer"
    readout = "max"
    dim = 0
    model = DGI(in_len, logit_len, gnn_layers=gnn_layers, readout_point=readout_point, readout=readout)
    output = model.readout(X, dim=dim)
    assert torch.equal(output, torch.max(X, dim=dim).values)


def test_DGI_readout_should_take_sum_of_embedding_if_readout_type_is_sum(dummy_graph):
    X, edge_index = dummy_graph
    in_len = X.shape[1]
    logit_len = 1
    gnn_layers = 3
    readout_point = "layer"
    readout = "sum"
    dim = 0
    model = DGI(in_len, logit_len, gnn_layers=gnn_layers, readout_point=readout_point, readout=readout)
    output = model.readout(X, dim=dim)
    assert torch.equal(output, torch.sum(X, dim=dim))


def test_DGI_forward_embedding_should_pass_embedding_into_gnn_layer_agg_if_readout_point_is_layer(dummy_graph):
    X, edge_index = dummy_graph
    in_len = X.shape[1]
    logit_len = 1
    gnn_layers = 3
    readout_point = "layer"
    model = DGI(in_len, logit_len, gnn_layers=gnn_layers, readout_point=readout_point)
    model_output = model.forward_embedding(X, edge_index)
    test_output = model.gnn_layer_agg(X, edge_index)
    assert torch.equal(model_output[0], test_output[0])
    assert torch.equal(model_output[1], test_output[1])


def test_DGI_forward_embedding_should_pass_embedding_into_gnn_output_agg_if_readout_point_is_output(dummy_graph):
    X, edge_index = dummy_graph
    in_len = X.shape[1]
    logit_len = 1
    gnn_layers = 3
    readout_point = "output"
    model = DGI(in_len, logit_len, gnn_layers=gnn_layers, readout_point=readout_point)
    model_output = model.forward_embedding(X, edge_index)
    test_output = model.gnn_output_agg(X, edge_index)
    assert torch.equal(model_output[0], test_output[0])
    assert torch.equal(model_output[1], test_output[1])


def test_DGI_corrupt_network_should_permute_features_randoly(dummy_graph):
    X, edge_index = dummy_graph
    in_len = X.shape[1]
    logit_len = 1
    gnn_layers = 3
    readout_point = "output"
    model = DGI(in_len, logit_len, gnn_layers=gnn_layers, readout_point=readout_point)
    corrupted = model.corrupt_network(X)
    assert not torch.equal(X, corrupted)
