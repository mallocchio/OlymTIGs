import numpy as np
import torch
from typing import List, Tuple

# Carica il modello precedentemente addestrato
model = torch.load('./model.pt')
model.eval()

# Funzione per caricare e predire un'immagine
def prediction(tensor):
    """
    Predice i numeri per una lista di immagini rappresentate come tensori.

    Args:
        tensor (List[torch.Tensor]): Lista di tensori contenenti le immagini da predire.

    Returns:
        List[np.ndarray, torch.Tensor, int]: Una lista di tuple contenenti:
            - np.ndarray: Probabilità predette per ciascun numero (logit).
            - torch.Tensor: Logaritmi delle probabilità predette.
            - int: Numero predetto per ciascuna immagine.
    """
    numero_immagini = len(tensor)
    predizioni = []
    for i in range(numero_immagini):
        predict = predict_number(tensor[i])
        predizioni.append(predict)

    return predizioni


# Funzione per effettuare la predizione
def predict_number(tensor: torch.Tensor) -> Tuple[np.ndarray, torch.Tensor, int]:
    """
    Predice un numero per un'immagine rappresentata come un tensore.

    Args:
        tensor (torch.Tensor): Il tensore contenente l'immagine da predire.

    Returns:
        Tuple[np.ndarray, torch.Tensor, int]: Una tupla contenente:
            - np.ndarray: Probabilità predette per ciascun numero.
            - torch.Tensor: Logaritmi delle probabilità predette.
            - int: Numero predetto per l'immagine.
    """
    tensor = tensor.view(1, 784)
    with torch.no_grad():
        logps = model(tensor)

    ps = torch.exp(logps)
    probab = list(ps.numpy()[0])
    predicted_digit = probab.index(max(probab))
    predizione = (ps, logps, predicted_digit)
    return predizione