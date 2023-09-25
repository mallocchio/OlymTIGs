from abc import ABC, abstractmethod
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

class Classifier(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def build_model(self):
        pass

    @abstractmethod
    def train(self, train_data, epochs=5):
        pass

    @abstractmethod
    def evaluate(self, test_data):
        pass

    @abstractmethod
    def save_model(self, model_path):
        pass

    @abstractmethod
    def load_model(self, model_path):
        pass

    @abstractmethod
    def preprocess_image(self, image):
        pass

    @abstractmethod
    def get_prediction(self, image):
        pass

    @abstractmethod
    def clean_prediction(self, ps, logps):
        pass

class TensorflowClassifier(Classifier):
    def __init__(self):
        super().__init__()
        self.num_classes = 10
        self.model = self.build_model()
        self.trained_model = None

    def build_model(self):
        model = models.Sequential([
            layers.Conv2D(6, 3, input_shape=(28, 28, 1), padding='same', activation='relu'),
            layers.MaxPool2D(),
            layers.Conv2D(3, 3, padding='same', activation='relu'),
            layers.MaxPool2D(),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self, train_ds, epochs=5):
        report = self.model.fit(train_ds, epochs=epochs)
        return report

    def evaluate(self, test_ds):
        return self.model.evaluate(test_ds)

    def save_model(self, model_path):
        self.model.save(model_path)

    # -----------------------------------------------

    def load_model(self, model_path):
        self.trained_model = tf.keras.models.load_model(model_path)

    def preprocess_image(self, image):
        tensor = image.resize((28, 28))
        tensor = np.array(tensor) / 255.0
        tensor = np.expand_dims(tensor, axis=0)
        return tensor

    def get_prediction(self, image):
        preprocessed_image = self.preprocess_image(image)
        ps = self.trained_model.predict(preprocessed_image)
        logps = np.log(ps / (1 - ps))
        prediction = self.clean_prediction(ps, logps)
        return prediction

    def clean_prediction(self, ps, logps):
        predicted_digit = np.argmax(logps[0])
        ps = ps.squeeze()
        logps = logps.squeeze()
        return ps, logps, predicted_digit


class TorchClassifier(Classifier):

    def __init__(self):
        super().__init__()
        self.input_size = 784
        self.hidden_sizes = [128, 64]
        self.output_size = 10
        self.model = self.build_model()
        self.criterion = nn.NLLLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.003, momentum=0.9)
        self.trained_model = None

    def build_model(self):
        model = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(self.hidden_sizes[0], self.hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(self.hidden_sizes[1], self.output_size),
            nn.LogSoftmax(dim=1)
        )
        return model

    def train(self, trainloader, epochs=15):
        for e in range(epochs):
            running_loss = 0
            for images, labels in trainloader:
                images = images.view(images.shape[0], -1)
                self.optimizer.zero_grad()
                output = self.model(images)
                loss = self.criterion(output, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            print("Epoch {} - Training loss: {}".format(e, running_loss / len(trainloader)))

    def evaluate(self, valloader):
        correct_count, all_count = 0, 0
        for images, labels in valloader:
            for i in range(len(labels)):
                img = images[i].view(1, 784)
                with torch.no_grad():
                    logps = self.model(img)
                ps = torch.exp(logps)
                probab = list(ps.numpy()[0])
                pred_label = probab.index(max(probab))
                true_label = labels.numpy()[i]
                if (true_label == pred_label):
                    correct_count += 1
                all_count += 1
        accuracy = correct_count / all_count
        return accuracy

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    # ---------------------------------------------

    def load_model(self, model_path):
        self.trained_model = torch.load(model_path)

    def preprocess_image(self, image):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        tensor = transform(image)
        tensor = tensor.view(1, 784)
        return tensor

    def get_prediction(self, image):
        preprocessed_image = self.preprocess_image(image)
        with torch.no_grad():
            logps = self.trained_model(preprocessed_image)
        ps = torch.exp(logps)
        prediction = self.clean_prediction(ps, logps)
        return prediction

    def clean_prediction(self, ps, logps):
        probab = list(ps.numpy()[0])
        predicted_digit = probab.index(max(probab))
        ps = ps.data.numpy().squeeze()
        logps = logps.data.numpy().squeeze()
        return ps, logps, predicted_digit

if __name__ == "__main__":
    # Codice di esempio per l'addestramento e l'uso dei classificatori Tensorflow e PyTorch
    # ...
    pass
