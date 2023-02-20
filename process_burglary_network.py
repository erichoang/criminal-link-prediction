"""
=================================== LICENSE ==================================
Copyright (c) 2021, Consortium Board ROXANNE
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

Neither the name of the ROXANNE nor the
names of its contributors may be used to endorse or promote products
derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY CONSORTIUM BOARD ROXANNE ``AS IS'' AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL CONSORTIUM BOARD TENCOMPETENCE BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
==============================================================================
"""

import json
import time
import torch
import ast
import networkx as nx
import numpy as np
import pandas as pd

from networkx.readwrite import json_graph
from torch_geometric.utils import from_networkx
from copy import deepcopy

def load_graph_from_json_file(graph_file_path):
    with open(graph_file_path) as json_file:
        data = json.load(json_file)
    G = json_graph.node_link_graph(data)
    return G

def reverse_array_from_string(str_array):
    arr = ast.literal_eval(str_array)
    # print(type(arr))
    return arr

def isNaN(num):
    return num != num

def remove_node_without_emb_case_summary(nx_graph):
    temp_graph = deepcopy(nx_graph)
    for node in temp_graph.nodes():
        if isNaN(temp_graph.nodes[node]['emb_case_summary']):
            nx_graph.remove_node(node)

def mask_node_by_year(nx_graph, pyg_graph, val_date='2019-1-1', test_date='2020-1-1'):
    count_train = 0
    count_val = 0
    count_test = 0
    train_mask = []
    val_mask = []
    test_mask = []
    for node in nx_graph.nodes():
        date = pd.to_datetime(nx_graph.nodes[node]['date'])
        date_val = pd.to_datetime(val_date)
        date_test = pd.to_datetime(test_date)
        if date < date_val:
            count_train += 1
            train_mask.append(True)
            val_mask.append(False)
            test_mask.append(False)
        elif date_val <= date < date_test:
            count_val += 1
            train_mask.append(False)
            val_mask.append(True)
            test_mask.append(False)
        elif date >= date_test:
            count_test += 1
            train_mask.append(False)
            val_mask.append(False)
            test_mask.append(True)
    
    print('count_train', count_train, count_train/len(nx_graph.nodes()))
    print('count_val', count_val, count_val/len(nx_graph.nodes()))
    print('count_test', count_test, count_test/len(nx_graph.nodes()))

    pyg_graph.train_mask = torch.BoolTensor(train_mask)
    pyg_graph.val_mask = torch.BoolTensor(val_mask)
    pyg_graph.test_mask = torch.BoolTensor(test_mask)


class BurglaryDataset:
    def __init__(self, root: str, name: str, val_date: str, test_date: str, device: torch.device):
        self.root = root
        self.dataset_name = name
        self.val_date = val_date
        self.test_date = test_date
        self.device = device
    
    def generate_pyg_graph(self):
        graph = load_graph_from_json_file(self.root)
        print(nx.info(graph))

        remove_node_without_emb_case_summary(graph)

        print(nx.info(graph))

        for node in graph.nodes():
            temp_list = reverse_array_from_string(graph.nodes[node]['emb_case_summary'])
            graph.nodes[node]['emb_case_summary'] = temp_list

        pyg_graph = from_networkx(graph, group_node_attrs=['emb_case_summary'], group_edge_attrs=['weight'])
        print(pyg_graph)

        mask_node_by_year(graph, pyg_graph, self.val_date, self.test_date)

        print(pyg_graph)

        return pyg_graph.to(self.device), self.dataset_name

if __name__ == "__main__":
    start = time.time()

    # Opening JSON file
    graph_file_path = 'dataset/israel_lea_inp_burglary/israel_lea_inp_burglary_v2_crime_id_network.json'
    graph = load_graph_from_json_file(graph_file_path)
    print(nx.info(graph))

    remove_node_without_emb_case_summary(graph)

    print(nx.info(graph))
    # print(len(graph.edges(data=True)))

    for node in graph.nodes():
        temp_list = reverse_array_from_string(graph.nodes[node]['emb_case_summary'])

        # graph.nodes[node]['emb_case_summary'] = torch.FloatTensor(temp_list)
        graph.nodes[node]['emb_case_summary'] = temp_list

    pyg_graph = from_networkx(graph, group_node_attrs=['emb_case_summary'], group_edge_attrs=['weight'])
    # pyg_graph = from_networkx(graph, group_edge_attrs=['weight'])
    print(pyg_graph)

    val_date = '2019-1-1'
    test_date = '2020-1-1'

    mask_node_by_year(graph, pyg_graph, val_date, test_date)

    print(pyg_graph)
    print(pyg_graph['edge_index'])
    # print(pyg_graph['original_edge'])
    # print(pyg_graph['original_node'])

    
    end = time.time()
    print('{:.4f} s'.format(end-start))
