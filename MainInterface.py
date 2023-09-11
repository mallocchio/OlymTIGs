import ImageGeneratorInterface
import GraphGeneratorInterface
import ImageToTensorConverterInterface
import ClassifierPredictionInterface
from enums import Type

# definisco il numero di immagini da generare

images = ImageGeneratorInterface.generaImmagini(10, Type.CREATED)

tensors = ImageToTensorConverterInterface.imageTensorConvert(images)

predictions = ClassifierPredictionInterface.prediction(tensors)

GraphGeneratorInterface.generateGraph(tensors, predictions)