from PIL import Image, ImageDraw, ImageFont
import numpy as np
from typing import List
import torch

def generaImmagini(numero_immagini: int) -> List[Image.Image]:
    """
    Genera un elenco di immagini casuali contenenti numeri disegnati.

    Args:
        numero_immagini (int): Il numero di immagini da generare.

    Returns:
        List[Image.Image]: Una lista di oggetti immagine contenenti numeri disegnati casualmente.
    """
    image_array = []

    # uso la funziione random per stampare i numeri in posiziioni diverse nella schermata 28*28
    for _ in range(numero_immagini):
        number = np.random.randint(0, 10)
        height = np.random.randint(4, 10)
        length = np.random.randint(-3, 3)
        fontSize = np.random.randint(22, 28)

        # Creare un'immagine vuota 28x28 con sfondo nero
        image = Image.new('L', (28, 28), color='black')

        # Creare un oggetto per disegnare sull'immagine
        draw = ImageDraw.Draw(image)

        # Disegnare il numero in bianco
        font = ImageFont.truetype('./arial.ttf', fontSize)
        draw.text((height, length), str(number), fill='white', font=font)

        # Aggiungi l'immagine alla lista
        image_array.append(image)

    return image_array


def generaImmaginiMnist(numero_immagini: int) -> List[Image.Image]:
    """
    Genera un elenco di immagini casuali prese dal dataset MNIST.

    Args:
        numero_immagini (int): Il numero di immagini da generare.

    Returns:
        List[Image.Image]]: Una lista di tensori contenenti le immagini MNIST.
    """
    image_array = []

    for _ in range(numero_immagini):
        index = np.random.randint(0, len(valset))
        image = valset[index]
        image_array.append(image[0])

    return image_array
