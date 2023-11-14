import cv2
import torchvision
import torchvision.datasets as datasets
from tensorflow.keras.datasets import mnist
import torch
import numpy as np
from torch.utils.data import Subset
from torchvision import transforms
import random

def prepare_dataset(label, TIG):

    if TIG == 'sinvad':

        test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)

        idx = test_dataset.targets==label
        idx = np.where(idx)[0]
        subset = Subset(test_dataset, idx)
        test_data_loader = torch.utils.data.DataLoader(dataset=subset, batch_size=1, shuffle=True)

        return test_data_loader
        

    if TIG == 'dlfuzz':

        (_, _), (x_test, y_test) = mnist.load_data()

        idxs = np.argwhere(y_test == label)
        dataset = x_test[idxs]

        random.shuffle(dataset)

        return dataset
