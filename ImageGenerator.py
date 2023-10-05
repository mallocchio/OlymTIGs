from abc import ABC, abstractmethod
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torchvision.datasets as datasets

class ImageGeneratorAbstract(ABC):

    def __init__(self):
        self.image = None

    @abstractmethod
    def create_image(self):
        pass

class ImageGenerator(ImageGeneratorAbstract):

    def __init__(self):
        super().__init__()
        self.number = None
        self.number_height = None
        self.number_length = None
        self.font_size = None

    def set_number(self):
        self.number = np.random.randint(0, 10)

    def set_number_height(self):
        self.number_height = np.random.randint(4, 10)

    def set_number_length(self):
        self.number_length = np.random.randint(-3, 3)

    def set_font_size(self):
        self.font_size = np.random.randint(22, 28)

    def create_image(self):
        self.set_number()
        self.set_number_height()
        self.set_number_length()
        self.set_font_size()

        self.image = Image.new('L', (28, 28), color='black')
        draw = ImageDraw.Draw(self.image)
        font = ImageFont.truetype('./arial.ttf', self.font_size)
        draw.text((self.number_height, self.number_length), str(self.number), fill='white', font=font)

        self.image = np.array(self.image)
        self.noisy_image = self.add_noise(self.image)

        return self.image, self.number

    def apply_filters(self, image):
        filtered_image = cv2.GaussianBlur(image, (5, 5), 0)
        filtered_image = cv2.Canny(image, 100, 200)
        return filtered_image

    def add_noise(self, image, noise_factor=0.5):
        noise = np.random.randn(*image.shape) * noise_factor
        noisy_image = image + noise
        noisy_image = np.clip(noisy_image, 0, 255)
        return noisy_image

class MNISTGenerator(ImageGeneratorAbstract):

    def __init__(self):
        super().__init__()
        self.valset = datasets.MNIST('TESTSET', download=True, train=False, transform=None)
        self.index = None
        self.label = None

    def set_index(self):
        self.index = np.random.randint(0, len(self.valset))

    def create_image(self):
        self.set_index()
        self.image, self.label = self.valset[self.index]
        self.image = np.array(self.image)
        return self.image, self.label

    def apply_filters(self, image):
        filtered_image = cv2.GaussianBlur(image, (5, 5), 0)
        filtered_image = cv2.Canny(image, 100, 200)
        return filtered_image

    def add_noise(self, image, noise_factor=0.5):
        noise = np.random.randn(*image.shape) * noise_factor
        noisy_image = image + noise
        noisy_image = np.clip(noisy_image, 0, 255)
        return noisy_image

if __name__ == "__main__":
    generator = MNISTGenerator()
    generator.create_image()
