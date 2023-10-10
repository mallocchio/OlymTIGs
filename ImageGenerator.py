import cv2
import numpy as np
import torchvision
import torchvision.datasets as datasets
import torch
from torch.utils.data import Subset
from torchvision import transforms

class ImageGenerator():

    def prepare_raw_data_loader(self, imgs_to_generate):
        valset = torchvision.datasets.MNIST(root='./data', train=False, transform=None, download=True)

        data_loader = []
        for i in range(imgs_to_generate):
            index = np.random.randint(0, len(valset))
            image, label = valset[index]
            image = np.array(image)
            data_loader.append((image, label))
        return data_loader

    def prepare_tensor_data_loader(self, label):
        test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)

        if label != -1:
            idx = test_dataset.targets == label
            subset = Subset(test_dataset, np.where(idx)[0])
            self.test_data_loader = torch.utils.data.DataLoader(dataset=subset, batch_size=1, shuffle=True)
        else:
            self.test_data_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)
        return self.test_data_loader

    def tranform_data_loader(self, dataloader):
        transformed_dataloader = []
        for image in dataloader:
            img = image[0]
            filtered_image = self.apply_filters(img)
            noisy_image = self.add_noise(filtered_image)
            transformed_dataloader.append((noisy_image, image[1]))
        return transformed_dataloader

    def apply_filters(self, image):
        filtered_image = cv2.GaussianBlur(image, (5, 5), 0)
        filtered_image = cv2.Canny(image, 100, 200)
        return filtered_image

    def add_noise(self, image, noise_factor=0.5):
        noise = np.random.randn(*image.shape) * noise_factor
        noisy_image = image + noise
        noisy_image = np.clip(noisy_image, 0, 255)
        return noisy_image
