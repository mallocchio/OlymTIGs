import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

class VisualizzatoreGrafico:

    def __init__(self, image, prediction):
        self.image = image
        self.prediction = prediction

    def visualizza_grafico(self):
        plt.style.use("ggplot")
        height = 0.2

        fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, figsize=(15, 8))

        ps = self.prediction[0]
        logit = self.prediction[1]

        max_ps = round(max(ps), 3)
        max_logit = round(max(logit), 3)
        predicted_digit = self.prediction[2]

        ax0.imshow(self.image, cmap='gray')
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