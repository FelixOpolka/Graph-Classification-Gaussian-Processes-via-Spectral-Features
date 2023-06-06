import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def mexican_hat_wavelet(l, beta=1.0):
    l = l * beta
    try:
        # return 2.0 * np.sqrt(2.0) / np.sqrt(3.0) * np.power(np.pi, -0.25) * (l / beta**2) * np.exp(-0.5 * (l / beta)**2)
        return 2.0 * np.sqrt(2.0 / 3.0) * np.power(np.pi, -1.0 / 4.0) * l ** 2 * np.exp(-0.5 * l ** 2)
    except FloatingPointError:
        return np.zeros_like(l)


def mexican_hat_wavelet_tf(l):
    const = tf.convert_to_tensor(2.0 * np.sqrt(2.0 / 3.0) * np.power(np.pi, -0.25))
    val = const * l**2 * tf.exp(-0.5 * l**2)
    return val


def low_pass_filter(l, alpha=5.0):
    return 1.0 / (1.0 + alpha * l)


def low_pass_filter_tf(l):
    return 1.0 / (1.0 + l)