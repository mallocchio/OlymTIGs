import numpy as np
from tensorflow.keras import backend as K
import tensorflow as tf
import os

from tensorflow.keras.datasets import mnist, fashion_mnist

from tensorflow.keras import losses

image_size = 28

# Logic for calculating reconstruction probability
def reconstruction_probability(dec, z_mean, z_log_var, X):
    """
    :param decoder: decoder model
    :param z_mean: encoder predicted mean value
    :param z_log_var: encoder predicted sigma square value
    :param X: input data
    :return: reconstruction probability of input
            calculated over L samples from z_mean and z_log_var distribution
    """
    sampled_zs = sampling([z_mean, z_log_var])
    mu_hat = dec(sampled_zs)

    reconstruction_loss = tf.reduce_mean(
        tf.reduce_sum(
            tf.keras.losses.binary_crossentropy(X, mu_hat), axis=(-1)
        )
    )

    return reconstruction_loss


# Calculates and returns probability density of test input
def calculate_density(x_target_orig, enc, dec):
    x_target_orig = np.clip(x_target_orig, 0, 1)
    x_target = np.reshape(x_target_orig, (-1, 28*28))
    z_mean, z_log_var, _ = enc(x_target)
    reconstructed_prob_x_target = reconstruction_probability(dec, z_mean, z_log_var, x_target)
    return reconstructed_prob_x_target

def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def reshape(x_test):
    x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
    x_test = x_test.astype('float32') / 255
    return x_test

def run_compute_rec_losses(encoder, decoder, dataset, run_folder):

    data = reshape(dataset)

    rec_losses = []

    for batch in data:
        batch = np.expand_dims(batch,0)
        rec_loss = calculate_density(batch, encoder, decoder)
        rec_losses.append(rec_loss)
    rec_loss_summary = np.vstack(rec_losses)

    return rec_loss_summary