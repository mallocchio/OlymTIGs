import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch import nn, optim
from time import time
import pandas as pd

class TorchMNISTClassifier:

    def __init__(self):
        self.input_size = 784
        self.hidden_sizes = [128, 64]
        self.output_size = 10
        self.model = self.build_model()
        self.criterion = nn.NLLLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.003, momentum=0.9)

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
        time0 = time()
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
        print("\nTraining Time (in minutes) =", (time() - time0) / 60)

    def evaluate_accuracy(self, valloader):
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
        torch.save(self.model, path)


class TrainedTorchClassifier:

    def __init__(self, model_path):
        self.model_path = model_path
        self.tensor = None
        self.trained_model = torch.load(self.model_path)

    def preprocess_image(self, image):
        transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                    ])
        self.tensor = transform(image)

    def get_prediction(self, image):
        self.preprocess_image(image)
        modified_tensor = self.tensor.view(1, 784)
        with torch.no_grad():
            logps = self.trained_model(modified_tensor)
        ps = torch.exp(logps)
        probab = list(ps.numpy()[0])
        predicted_digit = probab.index(max(probab))
        ps = ps.data.numpy()
        return self.tensor, ps, logps, predicted_digit

if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)), ])

    trainset = datasets.MNIST('PATH_TO_STORE_TRAINSET', download=True, train=True, transform=transform)
    valset = datasets.MNIST('PATH_TO_STORE_TESTSET', download=True, train=False, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)

    classifier = MNISTClassifier()
    classifier.train(trainloader)
    
    accuracy = classifier.evaluate_accuracy(valloader)
    print("Model Accuracy =", accuracy)
    
    classifier.save_model('./model.pt')
