import cv2
import torchvision
import torchvision.datasets as datasets
from tensorflow.keras.datasets import mnist
import torch
import numpy as np
from torch.utils.data import Subset
from torchvision import transforms
import random

'''UNIFICARE I 3 DATASET CREATOR IN MODO CHE CE NE SIA UNO SOLO CHE POI SCARICA QUELLO GIUSTO IN BASE AL MODELLO'''
'''DEVO IMPOSTARE LO STESSO MODO PER TUTTI, PER ESEMPIO ADESSO SINVAD USA UN DATASET MENTRE DLFUZZ UTILIZZA UN'ARRAY'''

def prepare_dataset(label, mode):

    if mode == 'sinvad':

        test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)

        if label != -1:
            idx = test_dataset.targets==label
            idx = np.where(idx)[0]
            subset = Subset(test_dataset, idx)
            test_data_loader = torch.utils.data.DataLoader(dataset=subset, batch_size=1, shuffle=True)
        else:
            test_data_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)

        print("Sinvad dataset created...\n")

        return test_data_loader
        

    if mode == 'dlfuzz':

        (_, _), (x_test, y_test) = mnist.load_data()

        if label != -1:
            idxs = np.argwhere(y_test == label)
            dataset = x_test[idxs]
        else:
            dataset = x_test

        random.shuffle(dataset)

        return dataset

    if mode == "test":

        valset = torchvision.datasets.MNIST(root='./data', train=False, transform=None, download=True)

        data_loader = []

        for i in range(imgs_to_generate):
            index = np.random.randint(0, len(valset))
            image, label = valset[index]
            image = np.array(image)
            data_loader.append((image, label))
        return data_loader
