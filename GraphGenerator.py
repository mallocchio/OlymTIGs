import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import List, Tuple

def crea_grafico(tensors, prediction):
    """
    Crea un grafico per ciascuna immagine e la relativa predizione.

    Args:
        tensors (List[torch.Tensor]): Lista di tensori contenenti le immagini.
        prediction (List[Tuple[np.ndarray, torch.Tensor, int]]): Lista di tuple contenenti le predizioni per ciascuna immagine.
            - np.ndarray: Probabilità predette per ciascun numero.
            - torch.Tensor: Logaritmi delle probabilità predette.
            - int: Numero predetto per l'immagine.

    Returns:
        None
    """
    num_tensors = len(tensors)

    for i in range(num_tensors):
        plt.style.use("ggplot")

        height = 0.2

        fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, figsize=(15, 8))

        ps = prediction[i][0].data.numpy().squeeze()
        logit = prediction[i][1].data.numpy().squeeze()

        max_ps = round(max(ps), 3)
        max_logit = round(max(logit), 3)
        predicted_digit = prediction[i][2]

        ax0.imshow(tensors[i].numpy().squeeze(), cmap='gray')
        ax0.axis('off')
        ax0.set_title(f'Il numero predetto è: {predicted_digit}')

        ax1.barh(np.arange(10), logit, height=height)
        ax1.set_yticks(np.arange(10))
        ax1.set_yticklabels(np.arange(10))
        ax1.set_title("Grafico dei logit")
        ax1.set_ylabel("Classi")
        ax1.set_xlabel(f'Logit più elevato: {max_logit:.3f}')

        ax2.barh(np.arange(10), ps, height=height)
        ax2.set_xlim(0, 1.1)
        ax2.set_yticks(np.arange(10))
        ax2.set_yticklabels(np.arange(10))
        ax2.set_title("Grafico delle predizioni")
        ax2.set_xlabel(f'Sicurezza della predizione: {max_ps*100:.3f}%')

        plt.tight_layout()
        plt.show()
