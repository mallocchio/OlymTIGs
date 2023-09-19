from abc import ABC, abstractmethod
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torchvision.datasets as datasets


class ImageGeneratorAbstract(ABC):

    def __init__(self):
        self.image = None
        self.tensor = None

    @abstractmethod
    def crea_immagine(self):
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

    def crea_immagine(self):
        self.set_number()
        self.set_number_height()
        self.set_number_length()
        self.set_font_size()

        self.image = Image.new('L', (28, 28), color='black')
        draw = ImageDraw.Draw(self.image)
        font = ImageFont.truetype('./arial.ttf', self.font_size)
        draw.text((self.number_height, self.number_length), str(self.number), fill='white', font=font)
        return self.image


class MNISTGenerator(ImageGeneratorAbstract):

    def __init__(self):
        super().__init__()
        self.valset = datasets.MNIST('TESTSET', download=True, train=False, transform=None)
        self.index = None

    def set_index(self):
        self.index = np.random.randint(0, len(self.valset))

    def crea_immagine(self):
        self.set_index()
        image, _ = self.valset[self.index]
        self.image = image
        return self.image