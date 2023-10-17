from Classifiers import TF_LeNet1, VAE, convert_tf_to_torch, Torch_LeNet1
from Gen_dataset import prepare_dlfuzz_dataset, prepare_sinvad_dataset, prepare_test_dataset
from Gen_bound_imgs import run_sinvad
from Gen_classifier_test import run_tester
from Gen_metis import run_dlfuzz
from FolderManager import *
import torch
from tensorflow.keras.layers import Input

class CompetitionInterface:
    def __init__(self, config_file):
        self.config_file = config_file
        self.load_config()

    def load_config(self):
        with open(self.config_file, 'r') as f:
            config = f.readlines()

        for line in config:
            key, value = map(str.strip, line.split(':'))
            
            if key == 'label':
                self.label = int(value)
            elif key == 'mode':
                self.mode = value
            elif key == 'vae_model_path':
                self.vae_model_path = value
            elif key == 'model_path':
                self.model_path = value
            elif key == 'results_path':
                self.results_path = value
            elif key == 'imgs_to_sample':
                self.imgs_to_sample = int(value)

    def train():
            
        TF_LeNet1(train=True)
        convert_tf_to_torch()

    def run_test(self):

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        torch_lenet1 = Torch_LeNet1()
        torch_lenet1.load_state_dict(torch.load(self.model_path))
        torch_lenet1.eval()
        torch_lenet1.to(device)

        dataset = prepare_test_dataset(self.imgs_to_sample)

        output_folder = create_folder(self.results_path, "test_run")

        run_tester(torch_lenet1, dataset, self.imgs_to_sample, output_folder)


    def run_gen(self):

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        ######CARICO IL VAE######
        vae = VAE(img_size=28 * 28, h_dim=1600, z_dim=400)
        vae.load_state_dict(torch.load(self.vae_model_path, map_location=device))
        vae.to(device)

        
        ######CARICO I MODELLI######
        torch_lenet1 = Torch_LeNet1()
        torch_lenet1.load_state_dict(torch.load(self.model_path))
        torch_lenet1.eval()
        torch_lenet1.to(device)
        print("model loaded...")

        
        img_rows, img_cols = 28, 28
        input_shape = (img_rows, img_cols, 1)
        input_tensor = Input(shape=input_shape)
        tf_lenet1 = TF_LeNet1(input_tensor=input_tensor)
                
        ######CARICO I DATASET######
        sinvad_dataset = prepare_sinvad_dataset(self.label)

        dlfuzz_dataset = prepare_dlfuzz_dataset(self.label)

    
        sinvad_run_folder = create_folder(self.results_path, "sinvad_run")

        dlfuzz_run_folder = create_folder(self.results_path, "dlfuzz_run")

        ######FACCIO PARTIRE GLI ALGORITMI######
        run_sinvad(self.label, device, vae, torch_lenet1, sinvad_dataset, self.imgs_to_sample, sinvad_run_folder)
        run_dlfuzz(self.label, tf_lenet1, input_tensor, dlfuzz_dataset, self.imgs_to_sample, dlfuzz_run_folder)


if __name__ == "__main__":
    competition = CompetitionInterface('Config.txt')
    
    if competition.mode == 'train':
        competition.train()
    elif competition.mode == 'run':
        competition.run_gen()
    elif competition.mode == 'test':
        competition.run_test()
    else:
        print('Error choosing mode')
    
    #competition.run_metis()
