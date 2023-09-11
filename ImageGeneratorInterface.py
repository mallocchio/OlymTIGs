from typing import List
from ImageGenerator import ImageGenerator
from enums import Type

def generaImmagini(num_images: int, type: Type) -> List[str]:
    """
    Genera un elenco di immagini in base al tipo specificato.

    Args:
        num_images (int): Il numero di immagini da generare.
        type (Type): Il tipo di generazione delle immagini, che può essere Type.MNIST o Type.CREATED.

    Returns:
        List[str]: Una lista di nomi di file delle immagini generate.

    Raises:
        ValueError: Se il tipo specificato non è valido.

    Note:
        - Per Type.MNIST, verranno generate immagini MNIST.
        - Per Type.CREATED, verranno generate immagini personalizzate.
    """
    images = []

    if type == Type.MNIST:
        images = ImageGenerator.generaImmaginiMnist(num_images)
    elif type == Type.CREATED:
        images = ImageGenerator.generaImmagini(num_images)
    else:
        raise ValueError("Tipo di generazione delle immagini non valido. Utilizzare Type.MNIST o Type.CREATED.")

    return images
