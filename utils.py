import random
import argparse
from collections import defaultdict
from typing import Iterable, Tuple, Union, Optional, List
import multiprocessing as mp

import networkx as nx
from sklearn.utils import resample
from sklearn.model_selection import KFold, StratifiedKFold
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from torch.nn.functional import one_hot
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, top_k_accuracy_score
from torch import Tensor
import torch
import numpy as np
import pandas as pd


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def to_numpy_cpu(*x: torch.Tensor) -> Union[list, torch.Tensor]:
    outputs = []
    for x_ in x:
        if isinstance(x_, torch.Tensor):
            x_ = x_.detach().cpu().numpy()
        outputs.append(x_)

    if len(outputs) == 1:
        return outputs[0]
    else:
        return outputs


def single_source_shortest_path_length_range(graph, node_range, cutoff):
    dists_dict = {}
    for node in node_range:
        dists_dict[node] = nx.single_source_shortest_path_length(graph, node, cutoff)
    return dists_dict


def merge_dicts(dicts):
    result = {}
    for dictionary in dicts:
        result.update(dictionary)
    return result


def all_pairs_shortest_path_length_parallel(graph, cutoff=None):
    nodes = list(graph.nodes)
    random.shuffle(nodes)
    num_workers = int(mp.cpu_count() * 0.8)
    pool = mp.Pool(processes=num_workers)
    results = [pool.apply_async(single_source_shortest_path_length_range,
            args=(graph, nodes[int(len(nodes)/num_workers*i):int(len(nodes)/num_workers*(i+1))], cutoff)) for i in range(num_workers)]
    output = [p.get() for p in results]
    dists_dict = merge_dicts(output)
    pool.close()
    pool.join()
    return dists_dict


def precompute_dist_data(edge_index, num_nodes, approximate=0) -> np.ndarray:
        '''
        Here dist is 1/real_dist, higher actually means closer, 0 means disconnected
        :return:
        '''
        graph = nx.Graph()
        edge_list = edge_index.transpose(1,0).tolist()
        graph.add_edges_from(edge_list)

        n = num_nodes
        dists_array = np.zeros((n, n))
        np.fill_diagonal(dists_array, 1)
        # dists_dict = nx.all_pairs_shortest_path_length(graph,cutoff=approximate if approximate>0 else None)
        # dists_dict = {c[0]: c[1] for c in dists_dict}
        dists_dict = all_pairs_shortest_path_length_parallel(graph,cutoff=approximate if approximate>0 else None)
        for i, node_i in enumerate(graph.nodes()):
            shortest_dist = dists_dict[node_i]
            for j, node_j in enumerate(graph.nodes()):
                dist = shortest_dist.get(node_j, -1)
                if dist!=-1:
                    # dists_array[i, j] = 1 / (dist + 1)
                    dists_array[node_i, node_j] = 1 / (dist + 1)

        return dists_array


def score_link_prediction(labels, scores):
    labels, scores = to_numpy_cpu(labels, scores)
    # return roc_auc_score(labels, scores), average_precision_score(labels, scores)
    return roc_auc_score(labels, scores), average_precision_score(labels, scores), top_k_accuracy_score(labels, scores, k=1)


def inductive_eval(cmodel, nodes, gt_labels, X, lambdas = (0, 1, 1)):
    # anode_emb = torch.sparse.mm(data.x, cmodel.attr_emb(torch.arange(data.x.shape[1]).to(cmodel.device)))
    test_data = Data(X, None)
    anode_emb = cmodel.attr_emb(test_data)

    first_embs = anode_emb[nodes[:, 0]]

    sec_embs = anode_emb[nodes[:, 1]]
    res = cmodel.attr_layer(first_embs, sec_embs) * lambdas[1]

    node_emb = anode_emb.clone()

    res = res + cmodel.inter_layer(first_embs, node_emb[nodes[:, 1]]) * lambdas[2]
    
    if len(res.shape) > 1:
        res = res.softmax(dim=1)[:, 1]

    res = res.detach().cpu().numpy()
    return score_link_prediction(gt_labels, res)


def transductive_eval(cmodel, edge_index, gt_labels, data, lambdas=(1, 1, 1)):
    res = cmodel.evaluate(edge_index, data, lambdas)
    if len(res.shape) > 1:
        res = res.softmax(dim=1)[:, 1]

    return score_link_prediction(gt_labels, res)


def detailed_eval(model,test_data,gt_labels,sp_M, evaluate,nodes_keep=None, verbose=False, lambdas=(1,1,1)):
    setting = {}

    setting['Full '] = lambdas
    setting['Inter'] = (0,0,1)
    if lambdas[1]:
        setting['Attr '] = (0,1,0)
    if lambdas[0]:
        setting['Node '] = (1,0,0)
    
    res = {}
    for s in setting:
        if not nodes_keep is None:
            if s != 'Node ':
                res[s] = evaluate(model, test_data, gt_labels,sp_M,nodes_keep,setting[s])
                if verbose:
                    print(s+' ROC-AUC:%.4f AP:%.4f'%res[s])
        else:            
            res[s] = evaluate(model, test_data, gt_labels,sp_M,setting[s])
            if verbose:
                print(s+' ROC-AUC:%.4f AP:%.4f'%res[s])
    return res


def seed_everything(seed: Optional[int] = None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def roxsd_to_dataset(graph: nx.Graph, node_features=None, edge_features=None):
    """
    Converts roxsd graph to torch geometric dataset
    :param graph: roxsdv2 graph
    """
    if isinstance(graph, nx.Graph):
        ng = nx.Graph()
    elif isinstance(graph, nx.DiGraph):
        ng = nx.DiGraph()
    else:
        raise Exception("No such graph")
    ng.add_nodes_from(graph.nodes)
    ng.add_edges_from(graph.edges)
    if not node_features:
        hot_embedding = pd.get_dummies([graph.degree[n] for n in ng.nodes]).to_numpy()
        for i, n in enumerate(ng.nodes):
            ng.nodes[n]["embedding"] = hot_embedding[i]
    dataset = from_networkx(ng, node_features if node_features else ["embedding"], edge_features)
    return dataset


class TransductiveGraphSampler:
    """ Class for sampling the graph for the transductive setting """

    def __init__(self, n_splits: int, stratified: str = None, random_state: int = None, shuffle: bool = False):
        """
        Initialize transductive graph sampler
        :param n_splits: number k of splits to make
        :param stratified: name of the node attribute that contains node classes for stratified sampling
        :param random_state: random seed for the sampler
        :param shuffle: should nodes be shuffled
        """
        self.n_splits = n_splits
        self.stratified = stratified
        self.random_state = random_state
        self.shuffle = shuffle

    def get_n_splits(self):
        """ Returns number of splits """
        return self.n_splits

    def _get_node_ids(self, graph):
        node_ids = []
        for id in graph.nodes:
            node_ids.append(id)
        return node_ids

    def _get_nodes_labels(self, graph, downsample_class, downsample_rate):
        stratified_nodes = {}
        for id, label in graph.nodes(data=self.stratified):
            if label not in stratified_nodes:
                stratified_nodes[label] = [id]
            else:
                stratified_nodes[label].append(id)
        if downsample_class:
            downsample_amount = int(len(stratified_nodes[downsample_class]) * downsample_rate)
            stratified_nodes[downsample_class] = resample(
                stratified_nodes[downsample_class],
                replace=False,
                n_samples=downsample_amount if downsample_amount else 1)
        nodes = []
        labels = []
        for c in stratified_nodes:
            for n in stratified_nodes[c]:
                nodes.append(n)
                labels.append(c)
        return (nodes, labels)

    def _kfold_sample(self, graph, sampler):
        node_ids = self._get_node_ids(graph)
        for train_idx, test_idx in sampler.split(graph.nodes):
            chosen_ids = (node_ids[i] for i in train_idx)
            yield (graph.subgraph(chosen_ids), graph)


    def _kfold_stratified_sample(self, graph, sampler, downsample_class, downsample_rate):
        nodes, labels = self._get_nodes_labels(graph, downsample_class, downsample_rate)
        test_subgraph = graph.subgraph(nodes)
        for train_idx, test_idx in sampler.split(nodes, labels):
            chosen_ids = (nodes[i] for i in train_idx)
            yield (graph.subgraph(chosen_ids), test_subgraph)


    def split(self, graph: nx.Graph, downsample_class: str = None, downsample_rate: float = 0.2) -> Iterable[Tuple[nx.Graph, nx.Graph]]:
        """
        yields split graphs for training and testing
        :param graph: networkx graph to be sampled
        :param downsample_class: name of the class that will be downsampled (attribute provided by stratified)
        :param downsample_rate: rate of downsapling [0. 1]
        :returns: Generator of (train_graph, test_graph) folds
        """
        sampler = None
        if self.stratified:
            sampler = StratifiedKFold(self.n_splits, random_state=self.random_state, shuffle=self.shuffle)
            return self._kfold_stratified_sample(graph, sampler, downsample_class, downsample_rate)
        else:
            sampler = KFold(self.n_splits, random_state=self.random_state, shuffle=self.shuffle)
            return self._kfold_sample(graph, sampler)


class InductiveGraphSampler(TransductiveGraphSampler):
    """ Sample graph for the inductive setting """

    def _kfold_sample(self, graph, sampler):
        node_ids = self._get_node_ids(graph)
        for train_idx, test_idx in sampler.split(graph.nodes):
            train_ids = (node_ids[i] for i in train_idx)
            test_ids = (node_ids[i] for i in test_idx)
            yield (graph.subgraph(train_ids), graph.subgraph(test_ids))

    def _kfold_stratified_sample(self, graph, sampler, downsample_class, downsample_rate):
        nodes, labels = self._get_nodes_labels(graph, downsample_class, downsample_rate)
        for train_idx, test_idx in sampler.split(nodes, labels):
            train_ids = (nodes[i] for i in train_idx)
            test_ids = (nodes[i] for i in test_idx)
            yield (graph.subgraph(train_ids), graph.subgraph(test_ids))
