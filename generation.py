def run_generation(TIG, model_name, model, run_folder, dataset, label, img_rows, img_cols, imgs_to_sample):

    if TIG == "sinvad":

        from tensorflow.keras.datasets import mnist
        from sinvad.vae import VAE, load_vae
        from sinvad.gen_bound_imgs import run_sinvad
        import torch

        vae_model_path = './trained/MNIST_EnD.pth'
        vae = load_vae(vae_model_path, img_rows, img_cols)
        run_sinvad(model_name, label, vae, model, dataset, img_rows, img_cols, imgs_to_sample, run_folder)

    if TIG == "dlfuzz":

        from tensorflow.keras.layers import Input
        from dlfuzz.gen_metis import run_dlfuzz
        from classifiers.classifiers import TF_LeNet1

        #expanding model of dlfuzz
        model, input_tensor = model

        run_dlfuzz(model_name, label, model, input_tensor, dataset, imgs_to_sample, run_folder)