import matplotlib.pyplot as plt
import numpy as np

class VisualizzatoreGrafico:

    def crea_grafico(self, image, prediction, image_with_noise, prediction_with_noise):
        plt.style.use("ggplot")
        height = 0.2

        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 12))  # Due righe di grafici

        def plot_image(ax, img, title, ylabel):
            ax.imshow(img, cmap='gray')
            ax.axis('off')
            ax.set_title(f'Il numero predetto è: {title}')
            ax.set_ylabel(ylabel)

        def plot_barh(ax, data, max_val, xlabel):
            ax.barh(np.arange(10), data, height=height)
            ax.set_yticks(np.arange(10))
            ax.set_yticklabels(np.arange(10))
            ax.set_title(xlabel)
            ax.set_xlabel(f'{xlabel}: {max_val:.3f}')

        plot_image(axes[0, 0], image, prediction[2], 'Immagine senza rumore')
        plot_barh(axes[0, 1], prediction[1], max(prediction[1]), 'Grafico dei logit')
        plot_barh(axes[0, 2], prediction[0], max(prediction[0]) * 100, 'Grafico delle predizioni')

        plot_image(axes[1, 0], image_with_noise, prediction_with_noise[2], 'Immagine con rumore')
        plot_barh(axes[1, 1], prediction_with_noise[1], max(prediction_with_noise[1]), 'Logit più elevato (con rumore)')
        plot_barh(axes[1, 2], prediction_with_noise[0], max(prediction_with_noise[0]) * 100, 'Sicurezza della predizione (con rumore)')

        plt.tight_layout()
        
        return fig

if __name__ == "__main__":
    # Esempio di utilizzo
    visualizzatore = VisualizzatoreGrafico()
    # Immagine senza rumore, predizione, Immagine con rumore, predizione con rumore
    # Sostituisci questi dati con i tuoi dati effettivi
    figura = visualizzatore.crea_grafico(image, prediction, image_with_noise, prediction_with_noise)
    plt.show()
