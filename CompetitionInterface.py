import os
import shutil
import time
import numpy as np
from Classifiers import TensorflowClassifier, TorchClassifier, VAE
from GraphGenerator import VisualizzatoreGrafico
from ImageGenerator import ImageGenerator
from Gen_bound_imgs import GeneticAlgorithm
from Gen_classifier_test import ClassifierTester
from Converter import ModelConverter
from FolderManager import FolderManager
import torch

class CompetitionInterface:
    def __init__(self, config_file):
        self.image_generator = ImageGenerator()
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
                self.folder_manager = FolderManager(self.results_path)
            elif key == 'imgs_to_sample':
                self.imgs_to_sample = int(value)

    def load_models(self):
        if self.model_path.endswith('.h5'):
            # Carica il classificatore TensorFlow
            self.classifier = TensorflowClassifier() 
            self.classifier.load_model(self.model_path)
        elif self.model_path.endswith('.pt'):
            # Carica il classificatore PyTorch
            self.classifier = TorchClassifier()
            self.classifier.load_model(self.model_path)
        else:
            raise ValueError('Model format not supported')

        print('Classifier initialized...')
        print('Model loaded...\n')

    def run_classifier_tester(self):
            
        self.dataset = self.image_generator.prepare_raw_data_loader(self.imgs_to_sample)
        self.transformed_dataset = self.image_generator.tranform_data_loader(self.dataset)

        output_folder = self.folder_manager.create_folder()

        self.load_models()

        ct = ClassifierTester(self.classifier, self.dataset, self.transformed_dataset, self.imgs_to_sample, output_folder)
        ct.run_tester()

    def run_bound_images_generator(self):

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.vae = VAE(img_size=28 * 28, h_dim=1600, z_dim=400)
        self.vae.load_state_dict(torch.load(self.vae_model_path, map_location=self.device))
        self.vae.to(self.device)

        if self.model_path.endswith('.h5'):
            print('Converting tensorflow classifier to torch...\n')
            percorso_originale = self.model_path
            nuovo_nome_file = "converted_model.pt"
            directory = os.path.dirname(percorso_originale)
            nuovo_percorso = os.path.join(directory, nuovo_nome_file)

            converter = ModelConverter(self.model_path, nuovo_percorso)
            tensorflow_classifier = converter.load_tensorflow_model()
            torch_classifier = converter.convert_to_pytorch(tensorflow_classifier)
            converter.save_pytorch_model(torch_classifier)
            self.model_path = nuovo_percorso
            print('Conversion ended successfully...\n')
                

        #carico i modelli
        self.load_models()

        # carico il dataset
        self.test_data_loader = ImageGenerator.prepare_tensor_data_loader(self, self.label)

        ga = GeneticAlgorithm(self.label, self.device, self.vae, self.classifier, self.test_data_loader, self.imgs_to_sample)

        results = ga.run_genetic_algorithm()

        output_folder = self.folder_manager.create_folder()
        ga.save_results(results, output_folder)

if __name__ == "__main__":
    competition = CompetitionInterface('Config.txt')
    if competition.mode == '1':
        competition.run_classifier_tester()
    elif competition.mode == '2':
        competition.run_bound_images_generator()
    else:
        print('Error choosing mode')
