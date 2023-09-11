import GraphGenerator
import torch
from typing import List, Tuple

def generateGraph(tensors: List[torch.Tensor], prediction: List[Tuple[np.ndarray, torch.Tensor, int]]) -> None:
    """
    Genera un grafico utilizzando la funzione 'GraphGenerator.crea_grafico' con i tensori e le predizioni forniti.

    Args:
        tensors (List[torch.Tensor]): Lista di tensori contenenti le immagini.
        prediction (List[Tuple[np.ndarray, torch.Tensor, int]]): Lista di tuple contenenti le predizioni per ciascuna immagine.
            - np.ndarray: Probabilità predette per ciascun numero.
            - torch.Tensor: Logaritmi delle probabilità predette.
            - int: Numero predetto per l'immagine.

    Returns:
        None
    """
    GraphGenerator.crea_grafico(tensors, prediction)
