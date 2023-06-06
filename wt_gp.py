"""
Wavelet-transform-based Gaussian process (WT-GP) that uses the energy captured by a wavelet-transformed signal to
distinguish graphs.
"""
import argparse
import math
import pathlib
import sys
import pickle as pk

import tensorflow as tf
import gpflow.kernels
import numpy as np
import pygsp as gsp
from gpflow import Parameter, default_float
import gpflow.covariances as cov
from gpflow.inducing_variables import InducingVariables
from gpflow.kernels import Matern12, Matern32, Polynomial, RBF
from gpflow.models import VGP, SVGP
from gpflow.utilities import positive
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from sklearn.metrics import accuracy_score, f1_score
from tensorflow.python.framework.ops import EagerTensor
from torch_geometric import datasets, loader

from synthetic_data import get_synthetic_dataset
from utils import create_dense_matrix_from_parts, compute_mean_results, \
    average_results_description_string, get_random_split, get_stratified_split, str2bool, get_node_degrees, \
    encode_one_hot
from wavelet_utils import mexican_hat_wavelet_tf, low_pass_filter_tf

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Spectrum Based Classification End2End')
    parser.add_argument('--data_dir', type=str, default="../data/", help='Path at which to store PyTorch Geometric datasets and look for precomputed files.')
    parser.add_argument('--dataset', type=str, default="ENZYMES", help='Name of the PyTorch Geometric dataset to evaluate on.')
    parser.add_argument('--num_runs', type=int, default=10, help='Number of runs to average over for performance evaluation.')
    parser.add_argument('--split', type=str, default="stratified", choices=["random", "stratified"], help='Type of data set split to use.')
    parser.add_argument('--train_on_val', type=str2bool, default=False, help='If True, the model is also trained on the validation set.')

    parser.add_argument('--num_scales', type=int, default=10, help='Number of learnable scales to use for Wavelets.')
    parser.add_argument('--max_initial_scale', type=float, default=5.0, help='Largest initial scale use for Wavelets.')
    parser.add_argument('--base_kernel', type=str, default="rbf", choices=["rbf", "matern12", "matern32", "poly"], help='Type of base kernel to use.')
    parser.add_argument('--num_bands', type=int, default=3, help='Number of band-pass filters to use for each wavelet filter.')

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    print(args)


class WaveletGraphClassificationKernel(gpflow.kernels.Kernel):
    def __init__(self, spectral_signals, eigenvalues, eigenvectors, train_idcs, num_scales=10, max_initial_scale=5.0,
                 num_bands=3, base_kernel="rbf"):
        super(WaveletGraphClassificationKernel, self).__init__()
        self.spectral_signals = spectral_signals
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        self.train_idcs = tf.reshape(tf.cast(train_idcs, tf.int32), -1)

        scales = np.random.uniform(low=0.1, high=max_initial_scale, size=[num_scales, num_bands])
        low_scales = np.random.uniform(low=4.0, high=6.0, size=[num_scales, 1])
        self.scales = Parameter(tf.convert_to_tensor(scales), dtype=default_float(), transform=positive())
        self.low_scale = Parameter(tf.convert_to_tensor(low_scales), dtype=default_float(), transform=positive())
        self.base_kernel = WaveletGraphClassificationKernel._get_kernel(base_kernel)

    @staticmethod
    def _get_kernel(kernel_name, num_dims=None, use_ard=False):
        if kernel_name == "rbf":
            return RBF() if not use_ard else RBF(lengthscales=tf.ones(num_dims, dtype=default_float()))
        elif kernel_name == "matern12":
            return Matern12() if not use_ard else Matern12(lengthscales=tf.ones(num_dims, dtype=default_float()))
        elif kernel_name == "matern32":
            return Matern32() if not use_ard else Matern32(lengthscales=tf.ones(num_dims, dtype=default_float()))
        elif kernel_name == "poly":
            return Polynomial() if not use_ard else Polynomial(variance=tf.ones(num_dims, dtype=default_float()))
        else:
            raise ValueError(f"Unknown kernel name {kernel_name}.")

    def _compute_unstd_graph_representations(self, idcs=None):
        eigenvalues = self.eigenvalues if idcs is None else [self.eigenvalues[idx] for idx in idcs]
        eigenvectors = self.eigenvectors if idcs is None else [self.eigenvectors[idx] for idx in idcs]
        spectral_signals = self.spectral_signals if idcs is None else [self.spectral_signals[idx] for idx in idcs]
        eigenvalues = WaveletGraphClassificationKernel._pad_tensors(eigenvalues)
        eigenvectors = WaveletGraphClassificationKernel._pad_tensors(eigenvectors)
        spectral_signals = WaveletGraphClassificationKernel._pad_tensors(spectral_signals)

        l_band = self.scales[None, :, :, None] * eigenvalues[:, None, None, :]              # [G, S, 3, N]
        g_band = mexican_hat_wavelet_tf(l_band)                                             # [G, S, 3, N]
        l_low = self.low_scale[None, :, :, None] * eigenvalues[:, None, None, :]            # [G, S, 1, N]
        g_low = low_pass_filter_tf(l_low)                                                   # [G, S, 1, N]
        g = tf.concat([g_low, g_band], axis=2)                                              # [G, S, 4, N]
        g = tf.reduce_sum(g, axis=2)                                                        # [G, S, N]
        g = tf.linalg.diag(g)                                                               # [G, S, N, N]
        w = tf.matmul(g, spectral_signals[:, None])                                         # [G, S, N, D]
        w = tf.matmul(eigenvectors[:, None], w)                                             # [G, S, N, D]
        w = tf.sqrt(tf.reduce_sum(tf.square(w), axis=2) + 1.0e-12)                          # [G, S, D]
        w = tf.reshape(w, [w.shape[0], -1])                                                 # [G, S*D]
        return w

    def compute_graph_representations(self, idcs=None):
        w_train = tf.stop_gradient(self._compute_unstd_graph_representations(self.train_idcs))
        w = self._compute_unstd_graph_representations(idcs)
        w = (w - tf.reduce_mean(w_train, axis=0)) / tf.math.reduce_std(w_train, axis=0)                 # [G, S*D]
        w = tf.where(tf.math.is_nan(w), tf.zeros_like(w), w)
        return w

    @staticmethod
    def _pad_tensors(list_of_arrays):
        max_dims = []
        for dim_idx in range(len(list_of_arrays[0].shape)):
            max_dims.append(max([array.shape[dim_idx] for array in list_of_arrays]))
        for dim_idx, max_dim in enumerate(max_dims):
            for idx in range(len(list_of_arrays)):
                list_of_arrays[idx] = np.concatenate([list_of_arrays[idx], np.zeros([*list_of_arrays[idx].shape[:dim_idx], max_dim-list_of_arrays[idx].shape[dim_idx], *list_of_arrays[idx].shape[dim_idx+1:]])], axis=dim_idx)
        padded = np.stack(list_of_arrays, axis=0)
        return tf.convert_to_tensor(padded)

    def K_uf(self, inducing_variable, X):
        Z = tf.reshape(inducing_variable.Z, inducing_variable.shape[:2])
        idcs = tf.reshape(tf.cast(X, tf.int32), -1)
        w = self.compute_graph_representations(idcs)
        cov = self.base_kernel(Z, w)
        return cov

    def K_uu(self, inducing_variable, jitter=None):
        Z = tf.reshape(inducing_variable.Z, inducing_variable.shape[:2])
        cov = self.base_kernel(Z, Z)
        if jitter is not None:
            cov = cov + tf.eye(cov.shape[0], dtype=cov.dtype) * jitter
        return cov

    def K(self, X, X2=None):
        idcs1 = tf.reshape(tf.cast(X, tf.int32), -1)
        w1 = self.compute_graph_representations(idcs1)                                       # [G', S*D]
        if X2 is not None:
            idcs2 = tf.reshape(tf.cast(X2, tf.int32), -1)
            w2 = self.compute_graph_representations(idcs2)
            cov = self.base_kernel(w1, w2)
        else:
            cov = self.base_kernel(w1, w1)
        return cov

    def K_diag(self, X):
        return tf.linalg.diag_part(self.K(X))


class GraphInducingVariable(InducingVariables):
    def __init__(self, initial_values):
        super(GraphInducingVariable, self).__init__()
        if len(initial_values.shape) == 2:
            initial_values = initial_values[..., None]
        self.Z = gpflow.Parameter(initial_values)

    @property
    def num_inducing(self):
        return self.Z.shape[0]

    @property
    def shape(self):
        return self.Z.shape


@cov.Kuu.register(GraphInducingVariable, WaveletGraphClassificationKernel)
def Kuu_spatial_additive_pariwise(inducing_variable, kernel, jitter=None):
    return kernel.K_uu(inducing_variable, jitter)


@cov.Kuf.register(GraphInducingVariable, WaveletGraphClassificationKernel, EagerTensor)
def Kuf_spatial_additive_pariwise(inducing_variable, kernel, X):
    return kernel.K_uf(inducing_variable, X)


def preprocess_data(ds, rstate, run_idx, train_on_val=False):
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
    fourier_signals, eigenvalues, eigenvectors, labels = [], [], [], []
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

        fourier_signal = U.T @ signal

        fourier_signals.append(fourier_signal)
        eigenvalues.append(l)
        eigenvectors.append(U)
        labels.append(batch.y)

    X = np.arange(len(fourier_signals)).reshape([-1, 1])
    y = np.array(labels).reshape([-1, 1])
    X_train = tf.convert_to_tensor(X[idcs_train], dtype=default_float())
    X_val = tf.convert_to_tensor(X[idcs_val], dtype=default_float())
    X_test = tf.convert_to_tensor(X[idcs_test], dtype=default_float())
    y_train = tf.convert_to_tensor(y[idcs_train], dtype=default_float())
    y_val = tf.convert_to_tensor(y[idcs_val], dtype=default_float())
    y_test = tf.convert_to_tensor(y[idcs_test], dtype=default_float())

    return fourier_signals, eigenvalues, eigenvectors, X_train, X_val, X_test, y_train, y_val, y_test


def get_trained_vgp_model(X_train, y_train, kernel, likelihood, num_latent_gps):
    model = VGP(data=(X_train, y_train), kernel=kernel, likelihood=likelihood, num_latent_gps=num_latent_gps)
    optimizer = gpflow.optimizers.Scipy()
    opt_result = optimizer.minimize(model.training_loss, model.trainable_variables, compile=False, step_callback=lambda step, b, c: print(f"{step}:\t{model.training_loss().numpy():.5f}\n" if step % 10 == 0 else "", end=""))
    return model, opt_result.fun


def get_trained_svgp_model(X_train, y_train, kernel, likelihood, num_latent_gps, num_inducing, batch_size=64,
                           verbose=True):
    # Use graph representations computed for subset of the training set and with randomly initialized scales to
    # initialize the inducing variables.
    inducing_idcs = tf.reshape(tf.cast(X_train[:num_inducing], tf.int32), -1)
    inducing_values = kernel.compute_graph_representations(inducing_idcs)
    inducing_variable = GraphInducingVariable(initial_values=inducing_values)
    model = SVGP(kernel=kernel, likelihood=likelihood, num_latent_gps=num_latent_gps,
                 inducing_variable=inducing_variable, num_data=X_train.shape[0])
    optimizer = tf.optimizers.Adam(learning_rate=0.01)

    # Training loop
    prev_loss, patience = None, 5
    for epoch in range(10**5):
        epoch_losses = []
        for start_idx in range(0, X_train.shape[0], batch_size):
            X_batch = X_train[start_idx:start_idx+batch_size]
            y_batch = y_train[start_idx:start_idx+batch_size]
            with tf.GradientTape() as tape:
                loss_value = model.training_loss(data=(X_batch, y_batch))
            grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            epoch_losses.append(loss_value.numpy())
        diff = prev_loss - np.mean(epoch_losses) if prev_loss is not None else None
        prev_loss = np.mean(epoch_losses)
        if verbose is True and epoch % 10 == 0:
            print(f"{epoch}:\tnELBO={prev_loss:.5f}")
        if diff is not None and diff < 0.1:
            patience -= 1
        if patience <= 0:
            if verbose is True: print("Early stopping")
            break
    return model, np.mean(epoch_losses)


def evaluate_for_hyperparameters(num_runs=10):
    results = []
    for run_idx in range(num_runs):
        result = evaluate_model(run_idx)
        results.append(result)
    mean_result, std_result = compute_mean_results(results)
    return mean_result, std_result


def evaluate_model(run_idx=0):
    if args.dataset in ["ENZYMES", "MUTAG", "NCI1", "IMDB-BINARY", "IMDB-MULTI"]:
        ds = datasets.TUDataset(name=args.dataset, root=args.data_dir, use_node_attr=True)
        evaluator = None
    elif args.dataset in ["ogbg-molhiv", "ogbg-molpcba", "ogbg-ppa", "ogbg-code2"]:
        ds = PygGraphPropPredDataset(name=args.dataset, root="../data/")
        evaluator = Evaluator(name=args.dataset)
    elif args.dataset in ["ring_clique", "sbm"]:
        ds = get_synthetic_dataset(args.dataset)
        evaluator = None
    else:
        raise NotImplementedError(f"No dataset called {args.dataset}.")
    use_sparse_model = args.dataset in ["ogbb-molhiv", "ogbg-molpcba", "ogbg-ppa", "ogbg-code2"]
    rstate = np.random.RandomState(seed=run_idx)

    spectral_signals, eigenvalues, eigenvectors, X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(
        ds, rstate, run_idx, train_on_val=args.train_on_val)
    num_classes = ds.num_classes

    kernel = WaveletGraphClassificationKernel(spectral_signals, eigenvalues, eigenvectors, train_idcs=X_train,
                                              num_scales=args.num_scales, max_initial_scale=args.max_initial_scale,
                                              num_bands=args.num_bands, base_kernel=args.base_kernel)
    likelihood = gpflow.likelihoods.MultiClass(num_classes) if num_classes > 2 else gpflow.likelihoods.Bernoulli()
    num_latent_gps = num_classes if num_classes > 2 else 1
    if use_sparse_model is True:
        batch_size = 32
        model, loss = get_trained_svgp_model(X_train, y_train, kernel, likelihood, num_latent_gps, num_inducing=100,
                                             batch_size=batch_size)
    else:
        model, loss = get_trained_vgp_model(X_train, y_train, kernel, likelihood, num_latent_gps)

    preds_val, preds_var_val = model.predict_y(X_val)
    preds_test, preds_var_test = model.predict_y(X_test)

    if evaluator is not None:   # If we are using an OGB data set
        preds_val, preds_test = preds_val.numpy(), preds_test.numpy()
        results_val = evaluator.eval({"y_true": y_val, "y_pred": preds_val})
        results_test = evaluator.eval({"y_true": y_test, "y_pred": preds_test})
        results_val = {"val_" + key: value for key, value in results_val.items()}
        results_test = {"test_" + key: value for key, value in results_test.items()}
        results = results_val | results_test
    else:
        preds_val, preds_test = preds_val.numpy(), preds_test.numpy()
        preds_val = preds_val.argmax(axis=-1) if preds_val.shape[-1] > 1 else (np.round(preds_val.reshape(-1))).astype(int)
        preds_test = preds_test.argmax(axis=-1) if preds_test.shape[-1] > 1 else (np.round(preds_test.reshape(-1))).astype(int)
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
    results["loss"] = loss
    print(results)
    return results


def run_evaluation():
    mean_result, std_result = evaluate_for_hyperparameters(num_runs=args.num_runs)
    print(args)
    print(average_results_description_string(mean_result, std_result))


if __name__ == '__main__':
    run_evaluation()