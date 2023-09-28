import matplotlib.pyplot as plt
import numpy as np
import time


class VisualizzatoreGrafico:

    def crea_grafico(self, image, prediction, image_with_noise, prediction_with_noise):
        plt.style.use("ggplot")
        height = 0.2

        fig, ((ax0, ax1, ax2), (ax3, ax4, ax5)) = plt.subplots(nrows=2, ncols=3, figsize=(15, 12))  # Due righe di grafici

        ps = prediction[0]
        logit = prediction[1]

        max_ps = round(max(ps), 3)
        max_logit = round(max(logit), 3)
        predicted_digit = prediction[2]

        ax0.imshow(image, cmap='gray')
        ax0.axis('off')
        ax0.set_title(f'Il numero predetto è: {predicted_digit}')
        ax0.set_ylabel('Immagine senza rumore')

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

        # Ripeti gli stessi grafici nella seconda riga

        noisy_ps = prediction_with_noise[0]
        noisy_logit = prediction_with_noise[1]
        noisy_predicted_digit = prediction_with_noise[2]

        max_noisy_ps = round(max(noisy_ps), 3)
        max_noisy_logit = round(max(noisy_logit), 3)

        ax3.imshow(image_with_noise, cmap='gray')
        ax3.axis('off')
        ax3.set_title(f'Il numero predetto è: {noisy_predicted_digit}')
        ax3.set_ylabel('Immagine senza rumore')

        ax4.barh(np.arange(10), noisy_logit, height=height)
        ax4.set_yticks(np.arange(10))
        ax4.set_yticklabels(np.arange(10))
        ax4.set_ylabel("Classi")
        ax4.set_xlabel(f'Logit più elevato: {max_noisy_logit:.3f}')

        ax5.barh(np.arange(10), noisy_ps, height=height)
        ax5.set_xlim(0, 1.1)
        ax5.set_yticks(np.arange(10))
        ax5.set_yticklabels(np.arange(10))
        ax5.set_xlabel(f'Sicurezza della predizione: {max_noisy_ps*100:.3f}%')

        plt.tight_layout()  # Aggiungi questa linea per garantire che i grafici siano ben posizionati


        return fig

        #plt.tight_layout()
        #plt.show()