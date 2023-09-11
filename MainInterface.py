import ImageGeneratorInterface
import GraphGeneratorInterface
import ImageToTensorConverterInterface
import ClassifierPredictionInterface
from enums import Type

# definisco il numero di immagini da generare
def main():
    numero_immagini = 10

    images = ImageGeneratorInterface.generaImmagini(numero_immagini, Type.CREATED)

    tensors = ImageToTensorConverterInterface.imageTensorConvert(images)

    predictions = ClassifierPredictionInterface.prediction(tensors)

    GraphGeneratorInterface.generateGraph(tensors, predictions)

if __name__ == "__main__":
    main()