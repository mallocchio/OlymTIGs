import glob
import math
import csv
import ntpath
import os

import numpy as np
import random
from tensorflow.keras import backend as K
import tensorflow as tf

image_size = 28

def run_validity_check(encoder, decoder, run_folder, label, thresholds):

    #VAE density threshold for classifying invalid inputs

    dlf_folder = run_folder + "/*.npy"
    filelist = [f for f in glob.glob(dlf_folder)]
    print("found samples: " + str(len(filelist)))

    for key, vae_threshold in thresholds.items():

        csv_file = os.path.join(run_folder, "ood_analysis_label_" + str(label) + "_th_" + str(key) + ".csv")

        with open(csv_file, 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['TOOL', 'SAMPLE', 'ID/OOD', 'loss'])

            for sample in filelist:
                s = np.load(sample)
                x_target = np.reshape(s, (-1, image_size*image_size))
                z_mean, z_log_var, _ = encoder(x_target)
                batch = K.shape(z_mean)[0]
                dim = K.int_shape(z_mean)[1]
                # by default, random_normal has mean = 0 and std = 1.0
                epsilon = K.random_normal(shape=(batch, dim))
                sampled_zs = z_mean + K.exp(0.5 * z_log_var) * epsilon
                mu_hat = decoder(sampled_zs)
                rec_loss = tf.reduce_mean(tf.reduce_sum(tf.keras.losses.binary_crossentropy(x_target, mu_hat), axis=(-1)))
                if rec_loss > vae_threshold or math.isnan(rec_loss):
                    distr = 'ood'
                else:
                    distr = 'id'
                loss = rec_loss.numpy()
                sample_name = ntpath.split(sample)[-1]
                writer.writerow(['DLF', sample_name, distr, loss])
            