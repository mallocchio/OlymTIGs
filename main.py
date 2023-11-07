import torch
import numpy as np
from gen_dataset import prepare_dataset
from folder_manager import create_folder

config_file = "Config.txt"

with open(config_file, 'r') as f:
    config = f.readlines()

    for line in config:
        if ":" in line:
            key, value = map(str.strip, line.split(':'))
            
            if key == 'label':
                label = int(value)
            elif key == 'mode':
                mode = value
            elif key == 'TIG':
                TIG = value
            elif key == 'model_name':
                model_name = value
            elif key == 'vae_model_path':
                vae_model_path = value
            elif key == 'model_path':
                model_path = value
            elif key == 'images_folder':
                images_folder = value
            elif key == 'results_path':
                results_path = value
            elif key == 'imgs_to_sample':
                imgs_to_sample = int(value)
            elif key == 'img_rows':
                img_rows =int(value)
            elif key == 'img_cols':
                img_cols = int(value)

if mode == "train":

    from classifiers.utils_classifiers import train_model

    train_model(model_name, img_rows, img_cols)

if mode == "generation":

    from classifiers.utils_classifiers import load_model

    run_folder = create_folder(results_path, TIG)
    dataset = prepare_dataset(label, TIG)

    if TIG == "sinvad":

        from sinvad.vae import VAE, load_vae
        from sinvad.gen_bound_imgs import run_sinvad

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        vae = load_vae(vae_model_path, img_rows, img_cols, device)
        torch_model = load_model(model_name, model_path, device=device)

        print("Models loaded...")

        run_sinvad(model_name, label, device, vae, torch_model, dataset, img_rows, img_cols, imgs_to_sample, run_folder)

    if TIG == "dlfuzz":

        from tensorflow.keras.layers import Input
        from dlfuzz.gen_metis import run_dlfuzz

        input_shape = (img_rows, img_cols, 1)
        input_tensor = Input(shape=input_shape)

        tf_model = load_model(model_name, model_path, input_tensor=input_tensor)

        run_dlfuzz(model_name, label, tf_model, input_tensor, dataset, imgs_to_sample, run_folder)

if mode == "validation":

    from selforacle.vae_chollet import load_encoder, load_decoder
    from selforacle.validity_check import run_validity_check

    encoder = load_encoder("trained/mnist_vae_all_classes/encoder")
    decoder = load_decoder("trained/mnist_vae_all_classes/decoder")

    run_validity_check(encoder, decoder, images_folder, label)
