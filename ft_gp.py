"""
Fourier-transform-based Gaussian process (FT-GP) that uses the cumulative energy functions to distinguish graphs.
"""
import argparse
import math
import pathlib
import sys
import pickle as pk

import numpy as np
import pygsp as gsp
from sklearn.metrics import accuracy_score, f1_score

from gaussian_process_classifier import GaussianProcessClassifier, SparseGaussianProcessClassifier
from synthetic_data import get_synthetic_dataset
from utils import create_dense_matrix_from_parts, compute_mean_results, \
    average_results_description_string, get_standardized_node_degrees, str2bool, get_stratified_split, get_random_split, \
    encode_one_hot, get_node_degrees

from torch_geometric import datasets
from torch_geometric import loader
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

from wavelet_utils import mexican_hat_wavelet, low_pass_filter

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Spectrum Based Classification')
    parser.add_argument('--data_dir', type=str, default="../data/", help='Path at which to store PyTorch Geometric datasets and look for precomputed files.')
    parser.add_argument('--dataset', type=str, default="ENZYMES", help='Name of the PyTorch Geometric dataset to evaluate on.')
    parser.add_argument('--num_runs', type=int, default=10, help='Number of runs to average over for performance evaluation.')
    parser.add_argument('--split', type=str, default="stratified", choices=["random", "stratified"], help='Type of data set split to use.')
    parser.add_argument('--train_on_val', type=str2bool, default=False, help='If True, the model is also trained on the validation set.')
    parser.add_argument('--train_frac', type=float, default=1.0, help='If not 1.0, use only the specified fraction of the training data for training.')
    parser.add_argument('--base_kernel', type=str, default="rbf", choices=["rbf", "matern12", "matern32", "poly"], help='Type of base kernel to use.')
    parser.add_argument('--num_eval_points', type=int, default=30, help='Number of linearly spaced evaluation points used to discretize the cumulative energy function.')

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    print(args)


def compute_cumulative_density(eigvals, eval_points, freq_signal):
    cum_signal = []
    for eval_point in eval_points:
        mask = (eigvals <= eval_point + 1e-7)
        cum_signal_val = np.sum(freq_signal[mask]**2, axis=0)
        cum_signal.append(cum_signal_val)
    cum_signal = np.stack(cum_signal, axis=0)
    return cum_signal


def get_eigenvalue_multiplicity(eigvals, tol=1e-5):
    eigvals = np.sort(eigvals)
    count = np.sum(np.abs(eigvals[1:] - np.roll(eigvals, shift=1)[1:]) < tol)
    return count


def get_cumsum_feature_vector(eigvecs, eigvals, signal, eval_points):
    eigenvalue_list = []
    eigenvalue_list.append(eigvals)
    freq_signal = eigvecs.T @ signal                                              # [N, D]
    cum_signals = []
    for eigvals in eigenvalue_list:
        cum_signal = compute_cumulative_density(eigvals, eval_points, freq_signal)
        cum_signals.append(cum_signal)
    cum_signal = np.concatenate(cum_signals, axis=1)

    # # for plot_idx in range(6, cum_signal.shape[-1]):
    # plt.xlabel("eigenvalue $\lambda$")
    # plt.ylabel("$e(z)$")
    # plt.plot(eval_points, cum_signal[:, -4], label="e(z)", linewidth=2.0)
    # plt.plot(eigvals, np.min(cum_signal[:, -4]) * np.ones_like(eigvals), "x", label="eigenvalues")
    # plt.legend()
    # plt.savefig("cumulative.png", bbox_inches="tight")
    # plt.show()

    return cum_signal


def get_dataset_split(ds, run_idx, train_on_val, rstate, train_frac=None):
    if hasattr(ds, "get_idx_split"):
        print("Using pre-defined split")
        split = ds.get_idx_split()
        idcs_train, idcs_val, idcs_test = split["train"], split["valid"], split["test"]
    elif args.split == "random":
        idcs_train, idcs_val, idcs_test = get_random_split(ds.data.y, rstate)
    elif args.split == "stratified":
        idcs_train, idcs_val, idcs_test = get_stratified_split(ds.data.y, run_idx, n_splits=args.num_runs)
    else:
        raise NotImplementedError(f"No split type called {args.split}.")

    if train_on_val is True:
        print("Also training on validation data.")
        idcs_train = np.concatenate([idcs_train, idcs_val])

    if train_frac is not None and train_frac != 1.0:
        print(f"Using only a fraction ({train_frac}) of the training data.")
        idcs_train = idcs_train[:int(train_frac * len(idcs_train))]

    return idcs_train, idcs_val, idcs_test


def preprocess_data(ds, eval_points, rstate, run_idx, train_on_val=False):
    """
    Takes a graph data set and produces spectrum-based features.
    :return:
        - inputs NumPy array of shape [N, K]
        - labels NumPy array of shape [N]
    """
    idcs_train, idcs_val, idcs_test = get_dataset_split(ds, run_idx=run_idx, train_on_val=args.train_on_val,
                                                        rstate=rstate, train_frac=args.train_frac)

    # Compute node degrees
    print("Computing node degrees statistics")
    data_loader = loader.DataLoader(ds[idcs_train], batch_size=1)
    node_degrees_list = []
    for batch in data_loader:
        graph_adj = create_dense_matrix_from_parts(
            batch.edge_index.numpy().T, num_nodes=batch.num_nodes,
            float_type=batch.edge_index.numpy().dtype)
        node_degrees = get_node_degrees(graph_adj)
        node_degrees_list.append(node_degrees)
    node_degrees = np.concatenate(node_degrees_list, axis=0)
    max_node_degree = math.ceil(np.percentile(node_degrees, q=99))
    print(f"Finished computing node degrees statistics: max_node_degree={max_node_degree}")

    # Construct spectral features
    np.seterr(divide='raise')
    data_loader = loader.DataLoader(ds, batch_size=1)
    inputs = []
    for batch in data_loader:
        graph_adj = create_dense_matrix_from_parts(
            batch.edge_index.numpy().T, num_nodes=batch.num_nodes,
            float_type=batch.edge_index.numpy().dtype)
        graph = gsp.graphs.Graph(graph_adj)
        try:
            graph.compute_laplacian("normalized")
        except FloatingPointError:
            # This will be a division by zero because of isolated nodes, but we can ignore it because it will just end
            # up being 0s in the Laplacian
            pass
        # node_degrees = get_standardized_node_degrees(graph_adj)
        node_degrees = get_node_degrees(graph_adj)
        node_degrees_oh = encode_one_hot(node_degrees, min_value=1, max_value=max_node_degree, as_numpy=True)
        if hasattr(batch, "x") and batch.x is not None:
            # signal = np.concatenate([node_degrees[:, None], batch.x.numpy()], axis=1)
            signal = np.concatenate([node_degrees_oh, batch.x.numpy()], axis=1)
        else:
            # signal = node_degrees[:, None]
            signal = node_degrees_oh
        L = graph.L.toarray()
        l, U = np.linalg.eigh(L)
        l[np.abs(l) < 1e-10] = 0.0
        U[np.abs(U) < 1e-10] = 0.0

        # # for plot_idx in range(batch.x.shape[-1]):
        # graph.set_coordinates()
        # plot_signal(graph, batch.x[:, -4].numpy(), plot_name="", save_as="graph.png")
        # plt.savefig("graph2.png", bbox_inches="tight")
        # plt.show()

        features = get_cumsum_feature_vector(eigvecs=U, eigvals=l, signal=signal, eval_points=eval_points)

        inputs.append(features)
    inputs = np.stack(inputs, axis=0)                                           # [N, P, D]
    inputs = np.reshape(inputs, [inputs.shape[0], -1])                          # [N, P*D]
    labels = ds.data.y.numpy()                                                  # [N]

    # Create training split
    X_train, X_val, X_test = inputs[idcs_train], inputs[idcs_val], inputs[idcs_test]
    y_train, y_val, y_test = labels[idcs_train], labels[idcs_val], labels[idcs_test]
    return X_train, y_train, X_val, y_val, X_test, y_test


def evaluate_runs(num_runs=10, print_result=True):
    results = []
    for run_idx in range(num_runs):
        result = evaluate_model(run_idx)
        results.append(result)
    mean_result, std_result = compute_mean_results(results)
    if print_result is True:
        print(args)
        print(average_results_description_string(mean_result, std_result))
    return mean_result, std_result


def evaluate_model(run_idx=0):
    if args.dataset in ["PROTEINS", "ENZYMES", "MUTAG", "NCI1", "DD", "IMDB-BINARY", "IMDB-MULTI"]:
        ds = datasets.TUDataset(name=args.dataset, root=args.data_dir, use_node_attr=True)
        evaluator = None
    elif args.dataset in ["ogbg-molhiv", "ogbg-molpcba", "ogbg-ppa", "ogbg-code2"]:
        ds = PygGraphPropPredDataset(name=args.dataset, root="../data/")
        evaluator = Evaluator(name=args.dataset)
    elif args.dataset in ["ring_clique", "erdos_renyi", "sbm", "path"]:
        ds = get_synthetic_dataset(args.dataset)
        evaluator = None
    else:
        raise NotImplementedError(f"No dataset called {args.dataset}.")
    rstate = np.random.RandomState(seed=run_idx)
    eval_points = np.linspace(0.0, 2.0, args.num_eval_points)

    X_train, y_train, X_val, y_val, X_test, y_test = preprocess_data(ds, eval_points, rstate, run_idx=run_idx,
                                                                     train_on_val=args.train_on_val)

    # model = DecisionTreeClassifier()
    model = GaussianProcessClassifier(kernel_name=args.base_kernel) if evaluator is None else SparseGaussianProcessClassifier(kernel_name=args.base_kernel)
    model.fit(X_train, y_train)
    if evaluator is not None:   # If we are using an OGB data set
        preds_val = model.predict(X_val, predict_classes=False)
        preds_test = model.predict(X_test, predict_classes=False)
        results_val = evaluator.eval({"y_true": y_val, "y_pred": preds_val})
        results_test = evaluator.eval({"y_true": y_test, "y_pred": preds_test})
        results_val = {"val_" + key: value for key, value in results_val.items()}
        results_test = {"test_" + key: value for key, value in results_test.items()}
        results = results_val | results_test
    else:
        preds_val, preds_var_val = model.predict(X_val, predict_classes=True)
        preds_test, preds_var_test = model.predict(X_test, predict_classes=True)
        acc_val = accuracy_score(y_val, preds_val)
        acc_test = accuracy_score(y_test, preds_test)
        f1_val = f1_score(y_val, preds_val, average="binary" if len(np.unique(y_val)) == 2 else "weighted")
        f1_test = f1_score(y_test, preds_test, average="binary" if len(np.unique(y_val)) == 2 else "weighted")
        results = {
            "acc_val": acc_val,
            "acc_test": acc_test,
            "f1_val": f1_val,
            "f1_test": f1_test,
        }
        # Store predictions
        with pathlib.Path(f"predictions_{args.dataset.lower()}_{run_idx}.pk").open("wb") as fd:
            pk.dump((preds_val, preds_var_val, y_val, preds_test, preds_var_test, y_test), fd)
    print(f"Results of run {run_idx}:", results)
    return results


if __name__ == '__main__':
    evaluate_runs(num_runs=args.num_runs)