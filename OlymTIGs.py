import sys
import torch
import numpy as np
import os
from gen_dataset import prepare_dataset
from folder_manager import create_folder
from classifiers.utils_classifiers import load_model, train_model
from generation import run_generation
from validation import run_validation

config_file = "Config.txt"

with open(config_file, 'r') as f:
    config = f.readlines()

    for line in config:
        if ":" in line:
            key, value = map(str.strip, line.split(':'))
            
            if key == 'label':
                label = int(value)
            elif key == 'mode':
                mode = value
            elif key == 'TIG':
                TIG = value
            elif key == 'model_path':
                model_path = value
                model_name = os.path.basename(model_path)
            elif key == 'images_folder':
                images_folder = value
            elif key == 'results_path':
                results_path = value
            elif key == 'imgs_to_sample':
                imgs_to_sample = int(value)
            elif key == 'img_rows':
                img_rows =int(value)
            elif key == 'img_cols':
                img_cols = int(value)
            elif key == 'validator':
                validator = value

if mode == "generation":

    run_folder = create_folder(TIG, results_path)
    dataset = prepare_dataset(label, TIG)

    if not os.path.exists(model_path):
        train_model(model_name, img_rows, img_cols)

    model = load_model(model_name, model_path, img_rows, img_cols)

    run_generation(TIG, model_name, model, run_folder, dataset, label, img_rows, img_cols, imgs_to_sample)

elif mode == "validation":

    run_validation(validator, images_folder)

else:
    "Modalità non presente"
