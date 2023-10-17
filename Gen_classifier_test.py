import os
import numpy as np
import time
import traceback
from Gen_dataset import tranform_image
from Gen_graph import crea_grafico
from Gen_dataset import *
from Torch_utils import *

def run_tester(classifier, dataset, imgs_to_sample, output_folder):

    correct_count = 0
    correct_count_with_noise = 0

    print('Start evaluating...\n')

    dataiter = iter(dataset)

    for i in range(imgs_to_sample):

        try:

            images, labels = next(dataiter)


            print(f"Processing image {i+1}")

            prediction = predict(classifier, images)

            if prediction[2] == labels:
                correct_count += 1


            modified_images = tranform_image(images)

            prediction_with_noise = predict(classifier, modified_images)

            if prediction_with_noise[2] == labels:
                correct_count_with_noise += 1
            
            fig = crea_grafico(images, prediction, modified_images, prediction_with_noise)

            graph_filename = os.path.join(output_folder, f"image_{i+1}.png")
            fig.savefig(graph_filename)

            np_image_filename = os.path.join(output_folder, f"image_{i+1}.npy")
            np.save(np_image_filename, modified_images)

        except Exception as e:
            # Registra l'eccezione invece di stamparla
            traceback.print_exc()

    print('Evaluation ended...\n')

    accuracy = correct_count / imgs_to_sample
    accuracy_with_noise = correct_count_with_noise / imgs_to_sample

    summary_file = os.path.join(output_folder, "summary.txt")

    with open(summary_file, 'w') as f:
        f.write(f"Classifier used: {classifier}\n")
        f.write(f"Images used: MNIST dataset\n")
        f.write(f"Images evaluated: {imgs_to_sample}\n")
        f.write(f"Accuracy of predictions without noise: {accuracy:.2%}\n")
        f.write(f"Accuracy of predictions with noise: {accuracy_with_noise:.2%}\n")
