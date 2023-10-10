import torch
import torch.nn as nn
from Classifiers import TensorflowClassifier

class PyTorchClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super(PyTorchClassifier, self).__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(6, 3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(3 * 7 * 7, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class ModelConverter:
    def __init__(self, tensorflow_model_path, pytorch_model_path):
        self.tensorflow_model_path = tensorflow_model_path
        self.pytorch_model_path = pytorch_model_path

    def load_tensorflow_model(self):
        tensorflow_classifier = TensorflowClassifier()
        tensorflow_classifier.load_model(self.tensorflow_model_path)
        return tensorflow_classifier

    def convert_to_pytorch(self, tensorflow_classifier):
        pytorch_classifier = PyTorchClassifier()
        
        def copy_weights(source, target):
            for source_layer, target_layer in zip(source.layers, target.children()):
                if isinstance(target_layer, nn.Conv2d) or isinstance(target_layer, nn.Linear):
                    target_layer.weight.data = torch.tensor(source_layer.get_weights()[0].transpose(3, 2, 0, 1))
                    target_layer.bias.data = torch.tensor(source_layer.get_weights()[1])

        copy_weights(tensorflow_classifier.trained_model, pytorch_classifier)
        return pytorch_classifier

    def save_pytorch_model(self, pytorch_classifier):
        torch.save(pytorch_classifier.state_dict(), self.pytorch_model_path)
