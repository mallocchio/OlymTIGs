import os
from selforacle.vae_chollet import load_encoder, load_decoder
from selforacle.validity_check import run_validity_check
import tensorflow
tensorflow.compat.v1.enable_eager_execution()

def run_validation(validator, images_folder):

    if validator == "selforacle":

        encoder_path = "./trained/validation_MNIST_vae/encoder"
        decoder_path = "./trained/validation_MNIST_vae/decoder"

        if not os.path.exists(encoder_path) or not os.path.exists(decoder_path):
            from selforacle.vae_chollet import train_vae
            train_vae()

        encoder = load_encoder(encoder_path)
        decoder = load_decoder(decoder_path)

        run_validity_check(encoder, decoder, images_folder)