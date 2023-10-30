import os
import numpy as np
import time
import traceback
from test.gen_graph import crea_grafico
from test.utils_test import *

def run_test(classifier, dataset, imgs_to_sample, output_folder):

    correct_count = 0
    correct_count_with_noise = 0

    print('Start evaluating...\n')

    start_time = time.time()

    dataiter = iter(dataset)

    for i in range(imgs_to_sample):

        try:

            images, labels = next(dataiter)


            print(f"Processing image {i+1}")

            prediction = predict(classifier, images)

            print(prediction[0])

            if prediction[2] == labels:
                correct_count += 1


            modified_images = tranform_image(images)

            prediction_with_noise = predict(classifier, modified_images)

            print(prediction_with_noise[0])

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

    end_time = time.time()

    accuracy = correct_count / imgs_to_sample
    accuracy_with_noise = correct_count_with_noise / imgs_to_sample

    summary_file = os.path.join(output_folder, "summary.txt")

    with open(summary_file, 'w') as f:
        f.write(f"Classifier used: {classifier}\n")
        f.write(f"Images used: MNIST dataset\n")
        f.write(f"Images evaluated: {imgs_to_sample}\n")
        f.write(f"Evaluation time: {end_time - start_time}\n")
        f.write(f"Accuracy of predictions without noise: {accuracy:.2%}\n")
        f.write(f"Accuracy of predictions with noise: {accuracy_with_noise:.2%}\n")
