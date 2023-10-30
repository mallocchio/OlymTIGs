import torch
import numpy as np

from sinvad.vae import VAE
from sinvad.gen_bound_imgs import run_sinvad
from gen_dataset import prepare_dataset
from folder_manager import create_folder
from selforacle.validity_check import run_validity_check
from selforacle.vae_chollet import load_encoder, load_decoder
from classifiers.classifiers import *
from classifiers.utils_classifiers import torch_load
from dlfuzz.gen_metis import run_dlfuzz
from tensorflow.keras.layers import Input

config_file = "Config.txt"

with open(config_file, 'r') as f:
    config = f.readlines()

    for line in config:
        key, value = map(str.strip, line.split(':'))
        
        if key == 'label':
            label = int(value)
        elif key == 'mode':
            mod = value
        elif key == 'model_name':
            model_name = value
        elif key == 'vae_model_path':
            vae_model_path = value
        elif key == 'model_path':
            model_path = value
        elif key == 'results_path':
            results_path = value
        elif key == 'imgs_to_sample':
            imgs_to_sample = int(value)


if mod == "train":

    if model_name == "lenet1":
        model = TF_LeNet1(train=True)
    elif model_name == "lenet4":
        model = TF_LeNet4(train=True)
    elif model_name == "lenet5":
        model = TF_LeNet5(train=True)

    convert_tf_to_torch(model, model_name)

run_folder = create_folder(results_path, mod)
dataset = prepare_dataset(imgs_to_sample, mod)

if mod == "test":

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch_lenet1 = Torch_LeNet1()
    torch_lenet1.load_state_dict(torch.load(model_path))
    torch_lenet1.eval()
    torch_lenet1.to(device)

    run_tester(torch_lenet1, dataset, imgs_to_sample, run_folder)

if mod == "sinvad":

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    vae = VAE(img_size=28 * 28, h_dim=1600, z_dim=400)
    vae.load_state_dict(torch.load(vae_model_path, map_location=device))
    vae.to(device)
    
    if model_name == "lenet1":
        torch_model = TF_LeNet1(train=True)
    elif model_name == "lenet4":
        torch_model = TF_LeNet4(train=True)
    elif model_name == "lenet5":
        torch_model = TF_LeNet5(train=True)

    torch_load(torch_model, model_path)
    print("Models loaded...")

    run_sinvad(label, device, vae, torch_model, dataset, imgs_to_sample, run_folder)

if mod == "dlfuzz":

    img_rows, img_cols = 28, 28
    input_shape = (img_rows, img_cols, 1)
    input_tensor = Input(shape=input_shape)

    if model_name == "lenet1":
        tf_model = TF_LeNet1(input_tensor=input_tensor, model_path=model_path)
    elif model_name == "lenet4":
        tf_model = TF_LeNet4(input_tensor=input_tensor, model_path=model_path)
    elif model_name == "lenet5":
        tf_model = TF_LeNet5(input_tensor=input_tensor, model_path=model_path)

    run_dlfuzz(label, tf_model, input_tensor, dataset, imgs_to_sample, run_folder)

encoder = load_encoder("trained/mnist_vae_all_classes/encoder")
decoder = load_decoder("trained/mnist_vae_all_classes/decoder")

#run_validity_check(encoder, decoder, run_folder)
run_validity_check(encoder, decoder, run_folder)
