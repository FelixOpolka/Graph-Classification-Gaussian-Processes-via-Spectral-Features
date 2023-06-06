import argparse
from collections import defaultdict
import numpy as np
import scipy.sparse as sp
from sklearn.model_selection import train_test_split, StratifiedKFold
import tensorflow as tf


def create_sparse_matrix_from_parts(adj_idcs, num_nodes, float_type, adj_vals=None):
    adj_vals = np.ones(adj_idcs.shape[0]) if adj_vals is None else adj_vals
    return sp.csr_matrix((adj_vals, (adj_idcs[:, 0], adj_idcs[:, 1])),
                         shape=(num_nodes, num_nodes), dtype=float_type)


def create_dense_matrix_from_parts(adj_idcs, num_nodes, float_type, adj_vals=None):
    adj_mat = create_sparse_matrix_from_parts(adj_idcs, num_nodes, float_type, adj_vals)
    return adj_mat.toarray()


def compute_mean_results(list_of_result_dicts):
    mean_results = defaultdict(list)
    for result_dict in list_of_result_dicts:
        for key, val in result_dict.items():
            mean_results[key].append(val)
    std_results = {}
    for key, val in mean_results.items():
        mean_results[key] = np.mean(val)
        std_results[key] = np.std(val)
    return mean_results, std_results


def average_results_description_string(mean_result, std_result):
    description = ""
    for key in mean_result.keys():
        description += f"{key}={mean_result[key]*100.0:.2f} +/- {std_result[key]*100.0:.2f}\t"
    return description


def get_standardized_node_degrees(adj):
    node_degrees = adj.sum(axis=1)
    node_degrees = node_degrees - np.mean(node_degrees)
    if np.std(node_degrees) > 1e-5:
        node_degrees = node_degrees / np.std(node_degrees)
    return node_degrees


def get_node_degrees(adj):
    node_degrees = adj.sum(axis=1)
    return node_degrees


def get_random_split(labels, rstate):
    idcs = np.arange(labels.shape[0])
    idcs_train, idcs_temp = train_test_split(idcs, train_size=0.8,
                                             shuffle=True, random_state=rstate)
    idcs_val, idcs_test = train_test_split(idcs_temp, train_size=0.5,
                                           shuffle=True, random_state=rstate)
    return idcs_train, idcs_val, idcs_test


def get_stratified_split(labels, split_idx, n_splits=10):
    rstate = np.random.RandomState(seed=0)
    num_data = labels.shape[0]
    fold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=rstate)
    splits = list(fold.split(np.zeros(num_data), labels.reshape(-1)))
    temp_idcs, test_idcs = splits[split_idx]
    train_idcs, val_idcs = train_test_split(temp_idcs, train_size=8.0/9.0, shuffle=True, random_state=rstate,
                                            stratify=labels[temp_idcs])
    assert len(np.intersect1d(train_idcs, val_idcs)) == 0
    assert len(np.intersect1d(train_idcs, test_idcs)) == 0
    assert len(np.intersect1d(val_idcs, test_idcs)) == 0
    return train_idcs, val_idcs, test_idcs


def encode_one_hot(values, min_value, max_value, as_numpy=True):
    indices = values.copy()
    indices[indices <= min_value] = min_value
    indices[indices >= max_value] = max_value
    encoding = tf.one_hot(values, depth=max_value-min_value)
    encoding = encoding if as_numpy is False else encoding.numpy()
    return encoding


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')