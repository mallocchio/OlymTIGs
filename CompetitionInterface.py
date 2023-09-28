import os
import time
import shutil
import numpy as np
from datetime import datetime
from Classifier import TensorflowClassifier, TorchClassifier
from GraphGenerator import VisualizzatoreGrafico
from ImageGenerator import ImageGenerator, MNISTGenerator

class CompetitionInterface:
    def __init__(self, config_file):
        self.config_file = config_file
        self.load_config()

    def load_config(self):
        with open(self.config_file, 'r') as f:
            config = f.readlines()

        for line in config:
            key, value = line.strip().split(':')
            key = key.strip()
            value = value.strip()

            if key == 'model_path':
                self.model_path = value
                if value.endswith('.h5'):
                    self.classifier = TensorflowClassifier()
                elif value.endswith('.pt'):
                    self.classifier = TorchClassifier()
                else:
                    raise ValueError("Classificatore non presente")
            elif key == 'num_images':
                self.num_images = int(value)
            elif key == 'images_type':
                if value == 'MNIST':
                    self.image_generator = MNISTGenerator()
                elif value == 'CREATED':
                    self.image_generator = ImageGenerator()
                else:
                    raise ValueError("Errore nella scelta del generatore di immagini: {}".format(value))

    def run(self):
        self.classifier.load_model(self.model_path)
        print('model loaded...')
        correct_count = 0
        correct_count_with_noise = 0

        # Genera un timestamp
        timestamp = int(time.time())

        # Inizializza output_folder
        output_folder = f"run_{timestamp}"
        os.makedirs(output_folder)

        for i in range(self.num_images):
            image = self.image_generator.create_image()
            print(f"Processing image {i + 1}")
            
            #try:
            prediction = self.classifier.get_prediction(image[0])

            if prediction[2] == image[1]:
                correct_count += 1

            image_with_filters = self.image_generator.apply_filters(image[0])
            image_with_noise = self.image_generator.add_noise(image_with_filters)
            prediction_with_noise = self.classifier.get_prediction(image_with_noise)

            if prediction_with_noise[2] == image[1]:
                correct_count_with_noise += 1
            
            # VisualizzatoreGrafico otterrà il timestamp direttamente da time.time()
            visualizer = VisualizzatoreGrafico()
            fig = visualizer.crea_grafico(image[0], prediction, image_with_noise, prediction_with_noise)

            # Salva il grafico in formato PNG con lo stesso timestamp nella cartella di output
            graph_filename = os.path.join(output_folder, f"{timestamp}_{i}.png")
            fig.savefig(graph_filename)

            # Salva l'immagine in formato numpy con lo stesso timestamp
            np_image_filename = os.path.join(output_folder, f"{timestamp}_{i}.npy")
            np.save(np_image_filename, image[0])

            time.sleep(1)

            #except Exception as e:
                #print(f"Errore durante l'elaborazione dell'immagine {i + 1}: {str(e)}")

        accuracy = correct_count / self.num_images
        accuracy_with_noise = correct_count_with_noise / self.num_images

        summary_file = os.path.join(output_folder, "summary.txt")
        with open(summary_file, 'w') as f:
            now = datetime.now()
            formatted_date_time = now.strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{formatted_date_time}\n")
            f.write(f"Classificatore usato: {self.classifier.__class__.__name__}\n")
            f.write(f"Immagini utilizzate: {self.image_generator.__class__.__name__}\n")
            f.write(f"Immagini valutate: {self.num_images}\n")
            f.write(f"Precisione delle predizioni senza rumore: {accuracy:.2%}\n")
            f.write(f"Precisione delle predizioni con rumore: {accuracy_with_noise:.2%}\n")

        for filename in os.listdir('.'):
            if filename.endswith(".png"):
                os.rename(filename, os.path.join(output_folder, filename))

        results_folder = "Results"
        os.makedirs(results_folder, exist_ok=True)
        shutil.move(output_folder, os.path.join(results_folder, output_folder))

if __name__ == "__main__":
    competition = CompetitionInterface('Config.txt')
    competition.run()
