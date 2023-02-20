import pytest
import networkx as nx
from utils import TransductiveGraphSampler, InductiveGraphSampler


@pytest.fixture
def network():
    g = nx.Graph()
    g.add_nodes_from([
        (0, {"class": "A"}),
        (1, {"class": "B"}),
        (2, {"class": "C"}),
        (3, {"class": "B"}),
        (4, {"class": "C"}),
        (5, {"class": "A"}),
        (6, {"class": "B"}),
        (7, {"class": "B"}),
        (8, {"class": "A"}),
        (9, {"class": "A"}),
        (10, {"class": "A"})
    ])
    g.add_edges_from([
        (0, 1),
        (1, 3),
        (2, 4),
        (1, 6),
        (0, 9),
        (8, 4),
        (8, 5),
        (8, 9),
        (7, 6),
        (0, 10)
    ])
    return g


def count_amounts(dist_dict, network, attr, norm=True):
    for node, a in network.nodes(data=attr):
        if a not in dist_dict:
            dist_dict[a] = 1
        else:
            dist_dict[a] += 1
    if norm:
        node_amount = network.number_of_nodes()
        for c in dist_dict:
            dist_dict[c] = dist_dict[c] / node_amount


def test_TransductiveGraphSampler_split_should_yield_split_amount_as_specified_by_n_splits(network):
    n_splits = 5
    sampler = TransductiveGraphSampler(n_splits)
    splits = [s for s in sampler.split(network)]
    assert len(splits) == n_splits


def test_TransductiveGraphSampler_split_should_yield_train_splits_that_exclude_test_nodes(network):
    n_splits = 3
    sampler = TransductiveGraphSampler(n_splits)
    nodes = set(network.nodes)
    edges = set(network.edges)
    for g_train, g_test in sampler.split(network):
        train_nodes = set(g_train.nodes)
        test_nodes = set(g_test.nodes)
        train_edges = set(g_train.edges)
        test_edges = set(g_test.edges)
        assert train_nodes.issubset(test_nodes)
        assert train_edges.issubset(train_edges)
        assert test_nodes - train_nodes
        assert test_edges - train_edges
        assert test_nodes == nodes
        assert test_edges == edges


def test_TransductiveGraphSampler_split_should_yield_splits_with_same_ratio_of_classes(network):
    n_splits = 3
    class_attr = "class"
    sampler = TransductiveGraphSampler(n_splits, class_attr)
    class_distribution = {}
    count_amounts(class_distribution, network, class_attr)
    for g_train, g_test in sampler.split(network):
        train_distribution = {}
        count_amounts(train_distribution, network, class_attr)
        for c in class_distribution:
            assert train_distribution[c] == pytest.approx(class_distribution[c])


def test_TransductiveGraphSampler_split_should_downsample_class_specified_in_downsample_class(network):
    n_splits = 2
    class_attr = "class"
    downsample_class = "A"
    downsample_rate = 0.5
    sampler = TransductiveGraphSampler(n_splits, class_attr)
    class_distribution = {}
    count_amounts(class_distribution, network, class_attr, False)
    class_distribution[downsample_class] = int(class_distribution[downsample_class] * downsample_rate)
    for g_train, g_test in sampler.split(network, downsample_class, downsample_rate):
        test_distribution = {}
        count_amounts(test_distribution, g_test, class_attr, False)
        assert test_distribution[downsample_class] == pytest.approx(class_distribution[downsample_class])


def test_InductiveGraphSampler_split_should_yield_subgraphs_of_original_graph_that_are_disjoint(network):
    n_splits = 3
    sampler = InductiveGraphSampler(n_splits)
    nodes = set(network.nodes)
    edges = set(network.edges)
    for g_train, g_test in sampler.split(network):
        train_nodes = set(g_train.nodes)
        test_nodes = set(g_test.nodes)
        assert train_nodes.issubset(nodes)
        assert test_nodes.issubset(nodes)
        assert train_nodes.isdisjoint(test_nodes)

def test_InductiveGraphSampler_split_should_yield_disjoint_stratified_subgraphs_if_stratified_specified(network):
    n_splits = 3
    class_attr = "class"
    sampler = InductiveGraphSampler(n_splits, class_attr)
    nodes = set(network.nodes)
    edges = set(network.edges)
    for g_train, g_test in sampler.split(network):
        train_nodes = set(g_train.nodes)
        test_nodes = set(g_test.nodes)
        assert train_nodes.issubset(nodes)
        assert test_nodes.issubset(nodes)
        assert train_nodes.isdisjoint(test_nodes)
