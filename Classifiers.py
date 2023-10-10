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
import cv2

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

    def load_sets(self):
        self.train_ds, self.test_ds = tfds.load(
            'mnist',
            split=['train', 'test'],
            as_supervised=True,
            batch_size=8  # Immagini alla volta che vengono passate
        )

    def train(self, epochs=5):
        report = self.model.fit(self.train_ds, epochs=epochs)
        return report

    def evaluate(self, test_ds):
        return self.model.evaluate(self.test_ds)

    def save_model(self, model_path):
        self.model.save(model_path)

    def load_model(self, model_path):
        self.trained_model = tf.keras.models.load_model(model_path)

    def preprocess_image(self, image):
        # Assicurati che l'immagine abbia una singola dimensione per il canale (1 per grayscale)
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=0)

        # Converte l'array numpy in un tensore TensorFlow
        image = tf.convert_to_tensor(image, dtype=tf.float32)
        return image

    def get_prediction(self, image):
        preprocessed_image = self.preprocess_image(image)
        ps = self.trained_model.predict(preprocessed_image)
        logps = np.log(ps / (1 - ps))
        prediction = self.clean_prediction(ps, logps)
        return prediction

    def clean_prediction(self, ps, logps):
        predicted_digit = np.argmax(ps[0])
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

    def load_sets(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        self.trainset = datasets.MNIST('PATH_TO_STORE_TRAINSET', download=True, train=True, transform=transform)
        self.valset = datasets.MNIST('PATH_TO_STORE_TESTSET', download=True, train=False, transform=transform)
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
        self.valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)

    def train(self, epochs=15):
        for e in range(epochs):
            running_loss = 0
            for images, labels in self.trainloader:
                images = images.view(images.shape[0], -1)
                self.optimizer.zero_grad()
                output = self.model(images)
                loss = self.criterion(output, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            print("Epoch {} - Training loss: {}".format(e, running_loss / len(self.trainloader)))

    def evaluate(self):
        correct_count, all_count = 0, 0
        for images, labels in self.valloader:
            for i in range(len(labels)):
                img = images[i].view(1, 784)
                with torch.no_grad():
                    logps = self.model(img)
                ps = torch.exp(logps)
                probab = list(ps.numpy()[0])
                pred_label = probab.index(max(probab))
                true_label = labels.numpy()[i]
                if true_label == pred_label:
                    correct_count += 1
                all_count += 1
        accuracy = correct_count / all_count
        return accuracy

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, model_path):
        self.trained_model = torch.load(model_path)

    def preprocess_image(self, image):
        # Converti l'array numpy in un tensore PyTorch
        tensor = torch.tensor(image, dtype=torch.float32)

        # Normalizza l'immagine
        tensor = (tensor / 255.0 - 0.5) / 0.5

        # Flatten l'immagine in modo che abbia le dimensioni (784,)
        tensor = tensor.view(1, 784)

        return tensor

    def get_prediction(self, image):
        preprocessed_image = self.preprocess_image(image)
        with torch.no_grad():
            logps = self.trained_model(preprocessed_image)
        ps = torch.exp(logps)
        prediction = self.clean_prediction(ps, logps)
        return prediction

    def get_batch_prediction(self, batch_images):
        all_logits = []
        for images in batch_images:
            image = images.view(1, 784)
            with torch.no_grad():
                logit = self.model(image)
                all_logits.append(logit)
        return all_logits

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



import torch.nn as nn
import torch.nn.functional as F

'''Simple VAE code'''

class ResBlock(nn.Module):
    def __init__(self, c_num):
        super(ResBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(c_num, c_num, 3, padding=1),
            nn.ReLU(),
            nn.InstanceNorm2d(c_num),
            nn.Conv2d(c_num, c_num, 3, padding=1),
            nn.ReLU(),
            nn.InstanceNorm2d(c_num)
        )   
    
    def forward(self, x):
        out = self.layer(x)
        out = x + out
        return out

class VAE(nn.Module):
    def __init__(self, img_size=28**2, h_dim=400, z_dim=50):
        super(VAE, self).__init__()
        self.z_dim = z_dim
        self.enc = nn.Sequential(
            nn.Linear(img_size, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, z_dim*2)
        )
        self.dec = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, img_size)
        )
        
    def encode(self, x):
        pre_z = self.enc(x)
        mu = pre_z[:, :self.z_dim]
        log_var = pre_z[:, self.z_dim:]
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        sigma = torch.exp(log_var/2)
        return mu + sigma * torch.randn(sigma.size()).cuda()
    
    def decode(self, z):
        return F.sigmoid(self.dec(z))
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconst = self.decode(z)
        return x_reconst, mu, log_var

class ConvVAE(nn.Module):
    def __init__(self, img_size=(28, 28), c_num=1, h_dim=3000, z_dim=400):
        super(ConvVAE, self).__init__()
        self.z_dim = z_dim
        self.img_h = img_size[0]
        self.enc_conv = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(c_num, 32, 3, padding=1),
                nn.ReLU(),
                nn.InstanceNorm2d(32)
            ),
            ResBlock(32),
            ResBlock(32),
            nn.MaxPool2d(2),
            nn.Sequential(
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.InstanceNorm2d(64)
            ),
            ResBlock(64),
            ResBlock(64),
            nn.MaxPool2d(2)
        )
        self.enc_linear = nn.Sequential(
            nn.Sequential(
                nn.Dropout(),
                nn.Linear((self.img_h**2)*64//16, h_dim),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Dropout(),
                nn.Linear(h_dim, 2*z_dim), # regularization only on linear
            ),
        )
        
        self.dec_linear = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(h_dim, (self.img_h**2)*64//16),
            nn.ReLU(),
        )
        self.dec_deconv = nn.Sequential(
            ResBlock(64),
            ResBlock(64),
            nn.Sequential(
                nn.ConvTranspose2d(64, 32, 4, 2, 1),
                nn.ReLU(),
                nn.InstanceNorm2d(32)
            ),
            ResBlock(32),
            ResBlock(32),
            nn.Sequential(
                nn.ConvTranspose2d(32, 16, 4, 2, 1),
                nn.ReLU(),
                nn.InstanceNorm2d(16)
            ),
            nn.Sequential(
                nn.Conv2d(16, c_num, 3, padding=1),
                nn.Sigmoid() # cause we want range to be in [0, 1]
            )
        )

    def encode(self, x):
        out = self.enc_conv(x)
        out = out.view(-1, (self.img_h**2)*64//16)
        pre_z = self.enc_linear(out)
        mu = pre_z[:, :self.z_dim]
        log_var = pre_z[:, self.z_dim:]
        return mu, log_var
    
    def decode(self, z):
        out = self.dec_linear(z)
        out = out.view(-1, 64, self.img_h//4, self.img_h//4)
        out = self.dec_deconv(out)
        return out
    
    def reparameterize(self, mu, log_var):
        sigma = torch.exp(log_var/2)
        return mu + sigma * torch.randn(sigma.size()).cuda()
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        rec_x = self.decode(z)
        return rec_x, mu, log_var
