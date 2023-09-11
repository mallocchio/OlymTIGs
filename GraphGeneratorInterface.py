import GraphGenerator
import torch
from typing import List, Tuple
import numpy

def generateGraph(tensors, prediction) -> None:
    """
    Genera un grafico utilizzando la funzione 'GraphGenerator.crea_grafico' con i tensori e le predizioni forniti.

    Args:
        tensors (List[torch.Tensor]): Lista di tensori contenenti le immagini.
        prediction (List[np.ndarray, torch.Tensor, int]): Lista di tuple contenenti le predizioni per ciascuna immagine.
            - np.ndarray: Probabilità predette per ciascun numero (logit).
            - torch.Tensor: Logaritmi delle probabilità predette.
            - int: Numero predetto per l'immagine.

    Returns:
        None
    """
    GraphGenerator.crea_grafico(tensors, prediction)
