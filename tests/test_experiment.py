#!/usr/bin/env python3
from torch_geometric.data import Data
from sklearn.model_selection import ParameterGrid
from dgi.dgi import DGI
from dgi.experiment import Experiment, Evaluation
from utils import InductiveGraphSampler, TransductiveGraphSampler
from unittest.mock import patch, MagicMock
import pytest
import random
import torch


experiment_parameters = {
    "experiment_name": ["test_experiment"],
    "seed": [123],
    "batch_size": [2, 4],
    "epochs": [400],
    "fold_amount": [10],
    "stratified_sample": [False],
    "downsample_class": [1],
    "downsample_rate": [0.1],
    "in_len": [10],
    "out_len": [2],
    "hidden_len": [64],
    "gin_hidden": [64],
    "gnn_arch": ["gcn"],
    "gnn_layers": [5],
    "normalize": ["batch", "layer"],
    "dropout_ratio": [0, 0.5],
    "activation": ["relu"],
    "readout": ["sum"],
    "readout_point": ["output"],
    "lof_neighbors": [20],
    "lof_leaf_size": [30],
    "lof_metric": ["minkowski"],
    "learning_rate": [0.0001],
    "weight_decay": [0.05],
    "validation_strategy": ["inductive_downsample"],
    "min_change": [0], # early stopping
    "change_metric": "accuracy_score",
    "patience": [15]
}

graph_amount = 20


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


@pytest.fixture
def dummy_dataset():
    graphs = []
    for i in range(graph_amount):
        graph = dummy_graph()
        label = torch.tensor(random.choice([0, 1]))
        g = Data(x=graph[0], edge_index=graph[1], y=label)
        graphs.append(g)
    return graphs


@pytest.fixture
def dummy_parameters():
    return ParameterGrid(experiment_parameters)


@pytest.fixture
def dummy_split_idx():
    idx = [i for i in range(graph_amount)]
    random.shuffle(idx)
    return {
        "train": idx[:10],
        "test": idx[10:15],
        "valid": idx[15:graph_amount]
    }


def test_Experiment_set_random_seed_should_set_random_seed_from_parameter_space(dummy_parameters, dummy_dataset):
    experiment = Experiment(dummy_parameters, dummy_dataset)
    assert experiment.seed == experiment_parameters["seed"][0]


def test_Experiment_get_experiment_should_return_initialized_model(dummy_parameters, dummy_dataset):
    experiment = Experiment(dummy_parameters, dummy_dataset)
    params = experiment.parameter_space[0]
    model, _ = experiment.get_experiment(params)
    assert isinstance(model, DGI)


def test_Experiment_get_experiment_should_return_initialized_optimizer(dummy_parameters, dummy_dataset):
    experiment = Experiment(dummy_parameters, dummy_dataset)
    params = experiment.parameter_space[0]
    model, optimizer = experiment.get_experiment(params)
    optimizer_state = optimizer.state_dict()
    parameter_amount = len([p for p in model.parameters()])
    assert len(optimizer_state["param_groups"][0]["params"]) == parameter_amount
    assert optimizer_state["param_groups"][0]["lr"] == params["learning_rate"]
    assert optimizer_state["param_groups"][0]["weight_decay"] == params["weight_decay"]


def test_Experiment_evaluate_should_iterate_over_parameter_space_and_yield_Evaluation(dummy_parameters, dummy_dataset):
    parameter_len = len(dummy_parameters)
    experiment = Experiment(dummy_parameters, dummy_dataset)
    eval_counter = 0
    for ev in experiment.evaluate():
        eval_counter += 1
        assert isinstance(ev, Evaluation)
        assert ev.params in dummy_parameters

# test splits

def test_Evaluation_train_should_train_model_for_one_epoch(dummy_parameters, dummy_dataset):
    experiment = Experiment(dummy_parameters, dummy_dataset)
    ev = next(experiment.evaluate())
    ev.set_splits()
    data = ev.splits[0]
    losses = ev.train(data[0])
    for l in losses:
        assert l > 0


def test_Evaluation_test_should_count_accuracy_and_f1_scores_for_model_outputs(dummy_parameters, dummy_dataset):
    experiment = Experiment(dummy_parameters, dummy_dataset)
    ev = next(experiment.evaluate())
    ev.set_splits()
    data = ev.splits[0]
    evaluations = ev.test(data[1])
    assert evaluations["accuracy_score"] >= 0 and evaluations["accuracy_score"] <= 1
    assert evaluations["f1_score"] >= 0 and evaluations["f1_score"] <= 1


def test_Evaluation_evaluate_setting_should_yield_starting_time_report_on_first_iteration(dummy_parameters, dummy_dataset):
    experiment = Experiment(dummy_parameters, dummy_dataset)
    ev = next(experiment.evaluate())
    report, params = next(ev.evaluate_setting())
    assert report["epoch"] == 0
    assert report.get("start_timestamp")


def test_Evaluation_evaluate_setting_should_yield_iteration_report_on_next_iterations(dummy_parameters, dummy_dataset):
    experiment = Experiment(dummy_parameters, dummy_dataset)
    ev = next(experiment.evaluate())
    settings = ev.evaluate_setting()
    next(settings)
    report, params = next(settings)
    d = ev.model.state_dict()
    assert "start_timestamp" in report
    assert "end_timestamp" in report
    assert len(report["batch_loss"].split(";"))
    assert "test_accuracy_score"
    assert "test_f1_score" in report
    assert all([torch.equal(params[p], d[p]) for p in d])


def test_Evaluation_evaluate_setting_should_yield_validation_scores_if_split_idx_set(dummy_parameters, dummy_dataset, dummy_split_idx):
    experiment = Experiment(dummy_parameters, dummy_dataset, dummy_split_idx)
    ev = next(experiment.evaluate())
    ev.params["validation_strategy"] = "standard_split"
    settings = ev.evaluate_setting()
    next(settings)
    report, params = next(settings)
    assert "valid_accuracy_score" in report
    assert "valid_f1_score" in report
