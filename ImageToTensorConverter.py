import torchvision.transforms as transforms
from PIL import Image
from typing import List
import torch

# Definisci la trasformazione per l'immagine
transform = transforms.Compose([
    # transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def imagesToTensor(images) -> List[torch.Tensor]:
    """
    Converte una lista di immagini PIL in una lista di tensori PyTorch.

    Args:
        images (List[Image.Image]): Una lista di immagini PIL.

    Returns:
        List[torch.Tensor]: Una lista di tensori PyTorch rappresentanti le immagini convertite.
    """
    numero_immagini = len(images)
    converted_images = []
    for i in range(numero_immagini):
        image = convert_image(images[i])
        converted_images.append(image)

    return converted_images

def convert_image(image: Image.Image) -> torch.Tensor:
    """
    Converte un'immagine PIL in un tensore PyTorch.

    Args:
        image (Image.Image): Un'immagine PIL.

    Returns:
        torch.Tensor: Un tensore PyTorch rappresentante l'immagine convertita.
    """
    image = image.convert('L')
    tensor = transform(image)
    # tensor = 255 - tensor_image # in caso l'immagine sia a colori invertiti
    return tensor
