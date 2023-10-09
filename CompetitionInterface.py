import os
import time
import shutil
import numpy as np
from datetime import datetime
from Classifier import TensorflowClassifier, TorchClassifier
from GraphGenerator import VisualizzatoreGrafico
from ImageGenerator import ImageGenerator, MNISTGenerator
from Gen_bound_imgs import GeneticAlgorithm
from model import VAE
from Converter import ModelConverter
import torch

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
                self.classifier = TensorflowClassifier() if value.endswith('.h5') else TorchClassifier()
                print(f'\n{self.classifier.__class__.__name__} classifier initialized...\n')
            elif key == 'imgs_to_sample':
                self.imgs_to_sample = int(value)
            elif key == 'images_type':
                if value == 'MNIST':
                    self.image_generator = MNISTGenerator()
                elif value == 'CREATED':
                    self.image_generator = ImageGenerator()
                else:
                    raise ValueError(f"Error in choosing image generator: {value}")

    def run_classifier_tester(self):
        self.classifier.load_model(self.model_path)
        print('Model loaded...\n')
        correct_count = 0
        correct_count_with_noise = 0

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_folder = f"run_{timestamp}"
        os.makedirs(output_folder)

        print('Start evaluating...\n')

        for i in range(self.imgs_to_sample):
            image = self.image_generator.create_image()
            print(f"Processing image {i + 1}")
            
            try:
                prediction = self.classifier.get_prediction(image[0])

                if prediction[2] == image[1]:
                    correct_count += 1

                image_with_filters = self.image_generator.apply_filters(image[0])
                image_with_noise = self.image_generator.add_noise(image_with_filters)
                prediction_with_noise = self.classifier.get_prediction(image_with_noise)

                if prediction_with_noise[2] == image[1]:
                    correct_count_with_noise += 1
                
                visualizer = VisualizzatoreGrafico()
                fig = visualizer.crea_grafico(image[0], prediction, image_with_noise, prediction_with_noise)

                graph_filename = os.path.join(output_folder, f"image_{i}.png")
                fig.savefig(graph_filename)

                np_image_filename = os.path.join(output_folder, f"image_{i}.npy")
                np.save(np_image_filename, image[0])

                time.sleep(1)

            except Exception as e:
                print(f"Error processing image {i + 1}: {str(e)}")

        print('Evaluation ended...\n')

        accuracy = correct_count / self.imgs_to_sample
        accuracy_with_noise = correct_count_with_noise / self.imgs_to_sample

        summary_file = os.path.join(output_folder, "summary.txt")

        with open(summary_file, 'w') as f:
            now = datetime.now()
            formatted_date_time = now.strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{formatted_date_time}\n")
            f.write(f"Classifier used: {self.classifier.__class__.__name__}\n")
            f.write(f"Images used: {self.image_generator.__class__.__name__}\n")
            f.write(f"Images evaluated: {self.imgs_to_sample}\n")
            f.write(f"Accuracy of predictions without noise: {accuracy:.2%}\n")
            f.write(f"Accuracy of predictions with noise: {accuracy_with_noise:.2%}\n")

        for filename in os.listdir('.'):
            if filename.endswith(".png"):
                os.rename(filename, os.path.join(output_folder, filename))

        results_folder = "Results"
        os.makedirs(results_folder, exist_ok=True)
        shutil.move(output_folder, os.path.join(results_folder, output_folder))

    def run_bound_images_generator(self):

        def load_models(self):

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
                self.classifier_model_path = nuovo_percorso
                print('Conversion ended successfully...\n')
                self.classifier = TorchClassifier()
                print('Classifier initialized...\n')

            elif self.model_path.endswith('.pt'):
                self.classifier = TorchClassifier()
                print('Classifier initialized...\n')

            self.classifier.load_model(self.model_path)
            print('Model loaded...\n')

        #carico i modelli
        load_models(self)

        ga = GeneticAlgorithm(self.label, self.device, self.vae, self.classifier, self.imgs_to_sample)
        ga.prepare_data_loader()
        results = ga.run_genetic_algorithm()

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_folder = f'run_{timestamp}'
        os.makedirs(run_folder)
        ga.save_results(results, run_folder)

        results_folder = "Bound_images_results"
        os.makedirs(results_folder, exist_ok=True)
        shutil.move(run_folder, os.path.join(results_folder, run_folder))

if __name__ == "__main__":
    competition = CompetitionInterface('Config.txt')
    if competition.mode == '1':
        competition.run_classifier_tester()
    elif competition.mode == '2':
        competition.run_bound_images_generator()
    else:
        print('Error choosing mode')
