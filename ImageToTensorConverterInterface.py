from typing import List
import torch
from ImageToTensorConverter import imagesToTensor

def imageTensorConvert(images: List) -> List[torch.Tensor]:
    """
    Converte una lista di immagini PIL in una lista di tensori PyTorch.

    Args:
        images (List): Una lista di immagini PIL.

    Returns:
        List[torch.Tensor]: Una lista di tensori PyTorch rappresentanti le immagini convertite.
    """
    tensor = imagesToTensor(images)
    return tensor
