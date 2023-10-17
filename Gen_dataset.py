import cv2
import torchvision
import torchvision.datasets as datasets
from tensorflow.keras.datasets import mnist
import torch
import numpy as np
from torch.utils.data import Subset
from torchvision import transforms
import random

def prepare_sinvad_dataset(label):
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)

    if label != -1:
        idx = test_dataset.targets==label
        idx = np.where(idx)[0]
        subset = Subset(test_dataset, idx)
        test_data_loader = torch.utils.data.DataLoader(dataset=subset, batch_size=1, shuffle=True)
    else:
        test_data_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)

    return test_data_loader


def prepare_dlfuzz_dataset(label):
    (_, _), (x_test, y_test) = mnist.load_data()

    if label != -1:
        idxs = np.argwhere(y_test == label)
        dataset = x_test[idxs]
    else:
        dataset = x_test

    random.shuffle(dataset)

    return dataset

def prepare_test_dataset(imgs_to_generate):
        valset = torchvision.datasets.MNIST(root='./data', train=False, transform=None, download=True)

        data_loader = []
        for i in range(imgs_to_generate):
            index = np.random.randint(0, len(valset))
            image, label = valset[index]
            image = np.array(image)
            data_loader.append((image, label))
        return data_loader

def tranform_image(image):
    img = apply_filters(image)
    img = add_noise(img)
    return img

def apply_filters(image):
    filtered_image = cv2.GaussianBlur(image, (5, 5), 0)
    filtered_image = cv2.Canny(image, 100, 200)
    return filtered_image

def add_noise(image, noise_factor=0.5):
    noise = np.random.randn(*image.shape) * noise_factor
    noisy_image = image + noise
    noisy_image = np.clip(noisy_image, 0, 255)
    return noisy_image
