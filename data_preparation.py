""" Prepare datasets for anomaly detection evaluation """
from typing import List, Tuple
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from process_burglary_network import load_graph_from_json_file, remove_node_without_emb_case_summary, reverse_array_from_string
from utils import TransductiveGraphSampler, InductiveGraphSampler
import pickle
import json
import networkx as nx
import pandas as pd

ROXSD_GROUNDTRUTH = "./dataset/roxsd_v2_core_graphs/graph_groundtruth_roxsd_v2.pkl"
ROXSD_GRAPH = "./dataset/roxsd_v2_core_graphs/graph_roxanne_roxsd_v2.pkl"

BURGLARY_GRAPH = "./dataset/israel_lea_inp_burglary/israel_lea_inp_burglary_v2_crime_id_network.json"

HIVE_GRAPH = "./dataset/hive_data/hive_graph_subscriptions_in_february.json"


def one_hot_attr(graph: nx.Graph):
    """
    Set one-hot embedding for each node based on degree
    :param graph: networkx graph
    """
    hot_embedding = pd.get_dummies([graph.degree[n] for n in graph.nodes]).to_numpy()
    for i, n in enumerate(graph.nodes):
       graph.nodes[n]["x"] = hot_embedding[i]


def prepare_roxsd(graph_path=ROXSD_GRAPH, groundtruth_path=ROXSD_GROUNDTRUTH) -> Data:
    """
    Read roxsd graph data and label anomalies
    :param graph_path: path to roxsd graph
    :parm groundtruth_path: path to groundruth roxsd path
    :returns: Data onject with labels
    """
    with open(graph_path, "rb") as f:
        graph = pickle.load(f)
    with open(groundtruth_path, "rb") as f:
        groundtruth_graph = pickle.load(f)
    for n in graph.nodes:
        if n not in groundtruth_graph.nodes:
            graph.nodes[n]["y"] = -1
        else:
            graph.nodes[n]["y"] = 1
    one_hot_attr(graph)
    return from_networkx(graph)


def prepare_burglary(n_splits: int, downsample_class: str, downsample_rate: float = 0.2, burglary_path=BURGLARY_GRAPH, sampling_mode="transductive", stratified="crime_type", random_state=None, shuffle=True) -> List[Tuple[Data, Data]]:
    """
    Read burglary dataset and label anomalies
    :param n_splits: Number of splits for sampler
    :param downsample_class: Class in stratified property to downsample
    :param downsample_rate: Fraction of the class that should be left
    :param burglary_path: Path to the burlgrary dataset path
    :param sampling_mode: 'transductive' or 'indeuctive' sampling mode
    :param n_splits: Number of splits to make
    :param stratified: name of the node attribute that contains node classes for stratified sampling
    :param random_state: random seed for the sampler
    :param shuffle: should nodes be shuffled
    :returns: Data object with labels
    """
    graph = load_graph_from_json_file(burglary_path)
    remove_node_without_emb_case_summary(graph)
    for node in graph.nodes():
        temp_list = reverse_array_from_string(graph.nodes[node]['emb_case_summary'])
        graph.nodes[node]["emb_case_summary"] = temp_list
    if sampling_mode == "transductive":
        sampler = TransductiveGraphSampler(n_splits, stratified, random_state, shuffle)
    elif sampling_mode == "inductive":
        sampler = InductiveGraphSampler(n_splits, stratified, random_state, shuffle)
    else:
        raise Exception("sampling_mode can be either 'transductive' or 'inductive'")
    burglary_splits = []
    for train, test in sampler.split(graph, downsample_class, downsample_rate):
        train_split = nx.Graph()
        test_split = nx.Graph()
        train_split.add_edges_from(train.edges())
        test_split.add_edges_from(test.edges())
        for n in train_split.nodes:
            train_split.nodes[n]["x"] = train.nodes[n]["emb_case_summary"]
            train_split.nodes[n]["y"] = -1 if train.nodes[n][stratified] == downsample_class else 1
        for n in test_split.nodes:
            test_split.nodes[n]["x"] = test.nodes[n]["emb_case_summary"]
            test_split.nodes[n]["y"] = -1 if test.nodes[n][stratified] == downsample_class else 1
        train_data = from_networkx(train_split)
        test_data = from_networkx(test_split)
        burglary_splits.append((train_data, test_data))
    return burglary_splits


def prepare_hive(hive_path=HIVE_GRAPH):
    """
    Read hive community subscriptions graph
    :param hive_path: path to the hive data json file
    """
    with open(hive_path, "r") as h:
        graph = json.load(h)
    constructed_graph = nx.Graph()
    constructed_graph.add_edges_from([(l["source"], l["target"]) for l in graph["links"]])
    for n in graph["nodes"]:
        constructed_graph.nodes[n["id"]]["y"] = 1 if n["type"] == "user" else -1
    one_hot_attr(constructed_graph)
    return from_networkx(constructed_graph)
