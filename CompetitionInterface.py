#import 
from Classifier import TensorflowClassifier, TorchClassifier
from GraphGenerator import VisualizzatoreGrafico
from ImageGenerator import ImageGenerator, MNISTGenerator

#scegliere se usare mnist oppure il creatore

class CompetitionInterface:

    def __init__(self, config_file):
        self.config_file = config_file
        self.load_config()

    def load_config(self):
        with open(self.config_file, 'r') as f:
            config = f.readlines()

        for line in config:
            key, value = line.strip().split(':')
            if key.strip() == 'model_path':
                self.model_path = value.strip()
            elif key.strip() == 'num_images':
                self.num_images = int(value.strip())
            elif key.strip() == 'images_type':
                value = value.strip()
                if value == 'MNIST':
                    self.image_generator = MNISTGenerator()
                elif value == 'CREATED':
                    self.image_generator = ImageGenerator()
                else:
                    raise ValueError("Errore nella scelta del generatore di immagini: {}".format(value))
            elif key.strip() == 'model':
                value = value.strip()
                if value == 'Torch':
                    self.classifier = TorchClassifier()
                elif value == 'Tensorflow':
                    self.classifier = TensorflowClassifier()
                else:
                    raise ValueError("Errore nella scelta del classificatore: {}".format(value))

    def run(self):
        self.classifier.load_model(self.model_path)
        for i in range(self.num_images):
            image = self.image_generator.crea_immagine()
            prediction = self.classifier.get_prediction(image)
            visualizer = VisualizzatoreGrafico(image, prediction)
            visualizer.visualizza_grafico()


if __name__ == "__main__":
    competition = CompetitionInterface('Config.txt')
    competition.run()