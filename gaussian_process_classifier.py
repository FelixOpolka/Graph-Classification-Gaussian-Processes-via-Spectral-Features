import numpy as np
import tensorflow as tf
import gpflow.likelihoods
from gpflow import default_float
from gpflow.inducing_variables import InducingPoints
from gpflow.kernels import RBF, Matern12, Matern32, Polynomial
from gpflow.models import VGP
from sklearn.preprocessing import StandardScaler


class GaussianProcessClassifier:
    def __init__(self, standardize=True, use_ard=False, kernel_name="rbf"):
        super(GaussianProcessClassifier, self).__init__()
        self.model = None
        self.standardizer = StandardScaler() if standardize is True else None
        self.use_ard = use_ard
        self.kernel_name = kernel_name

    def _get_kernel(self, num_dims=None):
        if self.kernel_name == "rbf":
            return RBF() if not self.use_ard else RBF(lengthscales=tf.ones(num_dims, dtype=default_float()))
        elif self.kernel_name == "matern12":
            return Matern12() if not self.use_ard else Matern12(lengthscales=tf.ones(num_dims, dtype=default_float()))
        elif self.kernel_name == "matern32":
            return Matern32() if not self.use_ard else Matern32(lengthscales=tf.ones(num_dims, dtype=default_float()))
        elif self.kernel_name == "poly":
            return Polynomial() if not self.use_ard else Polynomial(variance=tf.ones(num_dims, dtype=default_float()))
        else:
            raise ValueError(f"Unknown kernel name {self.kernel_name}.")

    def fit(self, X_train, y_train, num_classes=None):
        if self.standardizer is not None:
            X_train = self.standardizer.fit_transform(X_train)
        num_classes = len(np.unique(y_train)) if num_classes is None else num_classes
        X_train = tf.convert_to_tensor(np.reshape(X_train, [X_train.shape[0], -1]), dtype=default_float())
        y_train = tf.convert_to_tensor(np.reshape(y_train, [-1, 1]), dtype=default_float())
        kernel = self._get_kernel(X_train.shape[1])
        likelihood = gpflow.likelihoods.MultiClass(num_classes) if num_classes > 2 else gpflow.likelihoods.Bernoulli()
        num_latent_gps = num_classes if num_classes > 2 else 1
        self.model = VGP(data=(X_train, y_train), kernel=kernel, likelihood=likelihood, num_latent_gps=num_latent_gps)
        optimizer = gpflow.optimizers.Scipy()
        result = optimizer.minimize(self.model.training_loss, self.model.trainable_variables)

    def predict(self, X, predict_classes=True):
        if self.model is None:
            raise ValueError("Gaussian process model is untrained")
        if self.standardizer is not None:
            X = self.standardizer.transform(X)
        X = tf.convert_to_tensor(np.reshape(X, [X.shape[0], -1]), dtype=default_float())
        pred, pred_var = self.model.predict_y(X)
        pred, pred_var = pred.numpy(), pred_var.numpy()
        if predict_classes is True:
            if pred.shape[-1] == 1:
                pred = np.round(pred.reshape(-1)).astype(int)
            else:
                pred = np.argmax(pred, axis=-1)
        return pred, pred_var


class SparseGaussianProcessClassifier:
    def __init__(self, num_inducing_frac=0.05, batch_size=64, standardize=True, kernel_name="rbf", use_ard=False):
        super(SparseGaussianProcessClassifier, self).__init__()
        self.model = None
        self.standardizer = StandardScaler() if standardize is True else None
        self.num_inducing_frac = num_inducing_frac
        self.batch_size = batch_size
        self.kernel_name = kernel_name
        self.use_ard = use_ard

    def _get_kernel(self, num_dims=None):
        if self.kernel_name == "rbf":
            return RBF() if not self.use_ard else RBF(lengthscales=tf.ones(num_dims, dtype=default_float()))
        elif self.kernel_name == "matern12":
            return Matern12() if not self.use_ard else Matern12(lengthscales=tf.ones(num_dims, dtype=default_float()))
        elif self.kernel_name == "matern32":
            return Matern32() if not self.use_ard else Matern32(lengthscales=tf.ones(num_dims, dtype=default_float()))
        elif self.kernel_name == "poly":
            return Polynomial() if not self.use_ard else Polynomial(variance=tf.ones(num_dims, dtype=default_float()))
        else:
            raise ValueError(f"Unknown kernel name {self.kernel_name}.")

    def fit(self, X_train, y_train, num_classes=None):
        if self.standardizer is not None:
            X_train = self.standardizer.fit_transform(X_train)
        num_classes = len(np.unique(y_train)) if num_classes is None else num_classes
        X_train = tf.convert_to_tensor(np.reshape(X_train, [X_train.shape[0], -1]), dtype=default_float())
        y_train = tf.convert_to_tensor(np.reshape(y_train, [-1, 1]), dtype=default_float())
        kernel = self._get_kernel(X_train.shape[1])
        likelihood = gpflow.likelihoods.MultiClass(num_classes) if num_classes > 2 else gpflow.likelihoods.Bernoulli()

        inducing_variable = InducingPoints(X_train[:int(self.num_inducing_frac*X_train.shape[0])])
        num_latent_gps = num_classes if num_classes > 2 else 1
        self.model = gpflow.models.SVGP(kernel=kernel, likelihood=likelihood, inducing_variable=inducing_variable,
                                        num_latent_gps=num_latent_gps, num_data=X_train.shape[0])
        optimizer = tf.optimizers.Adam(learning_rate=0.01)

        prev_loss = None
        patience = 5
        for epoch in range(10**5):
            epoch_losses = []
            for start_idx in range(0, X_train.shape[0], self.batch_size):
                X_batch = X_train[start_idx:start_idx+self.batch_size]
                y_batch = y_train[start_idx:start_idx+self.batch_size]
                with tf.GradientTape() as tape:
                    loss_value = self.model.training_loss(data=(X_batch, y_batch))
                grads = tape.gradient(loss_value, self.model.trainable_variables)
                optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
                epoch_losses.append(loss_value.numpy())
            diff = prev_loss - np.mean(epoch_losses) if prev_loss is not None else None
            prev_loss = np.mean(epoch_losses)
            if epoch % 10 == 0:
                print(f"{epoch}:\tnELBO={prev_loss:.5f}")
            if diff is not None and diff < 0.1:
                patience -= 1
            if patience <= 0:
                print("Early stopping")
                break

    def predict(self, X, predict_classes=True):
        if self.model is None:
            raise ValueError("Gaussian process model is untrained")
        if self.standardizer is not None:
            X = self.standardizer.transform(X)
        X = tf.convert_to_tensor(np.reshape(X, [X.shape[0], -1]), dtype=default_float())
        pred, pred_var = self.model.predict_y(X)
        pred, pred_var = pred.numpy(), pred_var.numpy()
        if predict_classes is True:
            if pred.shape[-1] == 1:
                pred = (np.round(pred.reshape(-1))).astype(int)
            else:
                pred = np.argmax(pred, axis=-1)
        return pred
