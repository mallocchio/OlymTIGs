import os
import time
import shutil
import numpy as np
from datetime import datetime
from Classifier import TensorflowClassifier, TorchClassifier
from GraphGenerator import VisualizzatoreGrafico
from ImageGenerator import ImageGenerator, MNISTGenerator
from Gen_bound_imgs import GeneticAlgorithm
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
            elif key == 'num_images':
                self.num_images = int(value)
            elif key == 'images_type':
                if value == 'MNIST':
                    self.image_generator = MNISTGenerator()
                elif value == 'CREATED':
                    self.image_generator = ImageGenerator()
                else:
                    raise ValueError(f"Error in choosing image generator: {value}")

    def run(self):
        self.classifier.load_model(self.model_path)
        print('Model loaded...\n')
        correct_count = 0
        correct_count_with_noise = 0

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_folder = f"run_{timestamp}"
        os.makedirs(output_folder)

        print('Start evaluating...\n')

        for i in range(self.num_images):
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

        accuracy = correct_count / self.num_images
        accuracy_with_noise = correct_count_with_noise / self.num_images

        summary_file = os.path.join(output_folder, "summary.txt")
        with open(summary_file, 'w') as f:
            now = datetime.now()
            formatted_date_time = now.strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{formatted_date_time}\n")
            f.write(f"Classifier used: {self.classifier.__class__.__name__}\n")
            f.write(f"Images used: {self.image_generator.__class__.__name__}\n")
            f.write(f"Images evaluated: {self.num_images}\n")
            f.write(f"Accuracy of predictions without noise: {accuracy:.2%}\n")
            f.write(f"Accuracy of predictions with noise: {accuracy_with_noise:.2%}\n")

        for filename in os.listdir('.'):
            if filename.endswith(".png"):
                os.rename(filename, os.path.join(output_folder, filename))

        results_folder = "Results"
        os.makedirs(results_folder, exist_ok=True)
        shutil.move(output_folder, os.path.join(results_folder, output_folder))

    def run2(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        ga = GeneticAlgorithm(self.label, device, self.vae_model_path, self.model_path)
        ga.load_models()
        ga.prepare_data_loader()
        ga.run_genetic_algorithm()

if __name__ == "__main__":
    competition = CompetitionInterface('Config.txt')
    if competition.mode == '1':
        competition.run()
    elif competition.mode == '2':
        competition.run2()
    else:
        print('Error choosing mode')
