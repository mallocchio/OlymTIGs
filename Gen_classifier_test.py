import os
import numpy as np
import time
import traceback
from GraphGenerator import VisualizzatoreGrafico

class ClassifierTester():

    def __init__(self, classifier, dataset, transformed_dataset, imgs_to_sample, output_folder):
        self.classifier = classifier
        self.dataset = dataset
        self.transformed_dataset = transformed_dataset
        self.imgs_to_sample = imgs_to_sample
        self.output_folder = output_folder

    def run_tester(self):

        correct_count = 0
        correct_count_with_noise = 0

        print('Start evaluating...\n')

        for i in range(self.imgs_to_sample):

            try:

                print(f"Processing image {i+1}")
                image = self.dataset[i][0]
                label = self.dataset[i][1]

                prediction = self.classifier.get_prediction(image)

                if prediction[2] == label:
                    correct_count += 1

                transformed_image = self.transformed_dataset[i][0]
        
                prediction_with_noise = self.classifier.get_prediction(transformed_image)

                if prediction_with_noise[2] == label:
                    correct_count_with_noise += 1
                
                visualizer = VisualizzatoreGrafico()
                fig = visualizer.crea_grafico(image, prediction, transformed_image, prediction_with_noise)

                graph_filename = os.path.join(self.output_folder, f"image_{i+1}.png")
                fig.savefig(graph_filename)

                np_image_filename = os.path.join(self.output_folder, f"image_{i+1}.npy")
                np.save(np_image_filename, image[0])

            except Exception as e:
                # Registra l'eccezione invece di stamparla
                traceback.print_exc()

        print('Evaluation ended...\n')

        accuracy = correct_count / self.imgs_to_sample
        accuracy_with_noise = correct_count_with_noise / self.imgs_to_sample

        summary_file = os.path.join(self.output_folder, "summary.txt")

        with open(summary_file, 'w') as f:
            f.write(f"Classifier used: {self.classifier.__class__.__name__}\n")
            f.write(f"Images used: MNIST dataset\n")
            f.write(f"Images evaluated: {self.imgs_to_sample}\n")
            f.write(f"Accuracy of predictions without noise: {accuracy:.2%}\n")
            f.write(f"Accuracy of predictions with noise: {accuracy_with_noise:.2%}\n")
