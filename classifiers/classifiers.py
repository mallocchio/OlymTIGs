import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Input, Dense, Activation, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorflow.keras.layers import Input

def TF_LeNet1(input_tensor=None, train=False, model_path=None):
    nb_classes = 10
    # convolution kernel size
    kernel_size = (5, 5)

    if train:

        img_rows, img_cols = 28, 28
        input_shape = (img_rows, img_cols, 1)
        input_tensor = Input(shape=input_shape)

    elif input_tensor is None:
        print('you have to provide input_tensor when testing')
        exit()
    elif model_path is None:
        print('you have to provide model_path when testing')
        exit()

    # block1
    x = Conv2D(4, kernel_size, activation='relu', padding='same', name='block1_conv1')(input_tensor)
    x = MaxPool2D(pool_size=(2, 2), name='block1_pool1')(x)

    # block2
    x = Conv2D(12, kernel_size, activation='relu', padding='same', name='block2_conv1')(x)
    x = MaxPool2D(pool_size=(2, 2), name='block2_pool1')(x)

    x = Flatten(name='flatten')(x)
    x = Dense(nb_classes, name='before_softmax')(x)
    x = Activation('softmax', name='predictions')(x)

    model = Model(inputs=input_tensor, outputs=x)

    if train:
        
        trained_model = training(model)
        trained_model.save("./trained/lenet1.keras")
        
    else:

        model.load_weights(model_path)

    return model

class Torch_LeNet1(nn.Module):
    def __init__(self):
        kernel_size = (5, 5)
        super(Torch_LeNet1, self).__init__()
        # Block 1
        self.conv1 = nn.Conv2d(1, 4, kernel_size=kernel_size,
                               padding='same', stride=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2)

        # Block 2
        self.conv2 = nn.Conv2d(4, 12, kernel_size=kernel_size,
                               padding='same', stride=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(2)
        # 7x7x12 = 588
        self.out = nn.Linear(588, 10)

    def forward(self, x):
        out = self.maxpool1(self.relu1(self.conv1(x)))
        out = self.maxpool2(self.relu2(self.conv2(out)))
        out = np.transpose(out, (0, 2, 3, 1))
        out = torch.flatten(out, start_dim=1)
        #(10000, 588)
        out = self.out(out)  # [batch_size, 10]
        return out


def TF_LeNet4(input_tensor=None, train=False, model_path=None):
    nb_classes = 10
    # convolution kernel size
    kernel_size = (5, 5)

    if train:

        img_rows, img_cols = 28, 28
        input_shape = (img_rows, img_cols, 1)
        input_tensor = Input(shape=input_shape)

    elif input_tensor is None:
        print('you have to proved input_tensor when testing')
        exit()
    elif model_path is None:
        print('you have to proved model_path when testing')
        exit()

    # block1
    print("in Model2 input_tensor = ",input_tensor)
    x = Conv2D(6, kernel_size, activation='relu', padding='same', name='block1_conv1')(input_tensor)
    x = MaxPool2D(pool_size=(2, 2), name='block1_pool1')(x)

    # block2
    x = Conv2D(16, kernel_size, activation='relu', padding='same', name='block2_conv1')(x)
    x = MaxPool2D(pool_size=(2, 2), name='block2_pool1')(x)

    x = Flatten(name='flatten')(x)
    x = Dense(84, activation='relu', name='fc1')(x)
    x = Dense(nb_classes, name='before_softmax')(x)
    x = Activation('softmax', name='predictions')(x)

    model = Model(input_tensor, x)

    if train:
        
        trained_model = training(model)
        model.save_weights("./trained/lenet4.keras")
        
    else:

        model.load_weights(model_path)

    return model

class Torch_LeNet4(nn.Module):
    def __init__(self):
        kernel_size = (5, 5)
        super(Torch_LeNet4, self).__init__()
        # Block 1
        self.conv1 = nn.Conv2d(1, 6, kernel_size=kernel_size, padding=2)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2)

        # Block 2
        self.conv2 = nn.Conv2d(6, 16, kernel_size=kernel_size, padding=2)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(2)

        # Fully Connected Layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 7 * 7, 84)
        self.out = nn.Linear(84, 10)

    def forward(self, x):
        out = self.maxpool1(self.relu1(self.conv1(x)))
        out = self.maxpool2(self.relu2(self.conv2(out)))
        out = self.flatten(out)
        out = self.fc1(out)
        out = self.out(out)
        return out



def TF_LeNet5(input_tensor=None, train=False, model_path=None):
    nb_classes = 10
    # convolution kernel size
    kernel_size = (5, 5)

    if train:
        
        img_rows, img_cols = 28, 28
        input_shape = (img_rows, img_cols, 1)
        input_tensor = Input(shape=input_shape)

    elif input_tensor is None:
        print('you have to proved input_tensor when testing')
        exit()
    elif model_path is None:
        print('you have to proved model_path when testing')
        exit()

    # block1
    x = Convo2D(6, kernel_size, activation='relu', padding='same', name='block1_conv1')(input_tensor)
    x = MaxPool2D(pool_size=(2, 2), name='block1_pool1')(x)

    # block2
    x = Conv2D(16, kernel_size, activation='relu', padding='same', name='block2_conv1')(x)
    x = MaxPool2D(pool_size=(2, 2), name='block2_pool1')(x)

    x = Flatten(name='flatten')(x)
    x = Dense(120, activation='relu', name='fc1')(x)
    x = Dense(84, activation='relu', name='fc2')(x)
    x = Dense(nb_classes, name='before_softmax')(x)
    x = Activation('softmax', name='predictions')(x)

    model = Model(input_tensor, x)

    if train:
       
       trained_model = training(model)
       model.save_weights("./trained/lenet5.keras")

    else:
        model.load_weights(model_path)

    return model

class Torch_LeNet5(nn.Module):
    def __init__(self):
        kernel_size = (5, 5)
        super(Torch_LeNet3, self).__init__()
        # Block 1
        self.conv1 = nn.Conv2d(1, 6, kernel_size=kernel_size, padding=2)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2)

        # Block 2
        self.conv2 = nn.Conv2d(6, 16, kernel_size=kernel_size, padding=2)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(2)

        # Fully Connected Layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 7 * 7, 120)
        self.fc2 = nn.Linear(120, 84)
        self.out = nn.Linear(84, 10)

    def forward(self, x):
        out = self.maxpool1(self.relu1(self.conv1(x)))
        out = self.maxpool2(self.relu2(self.conv2(out)))
        out = self.flatten(out)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.out(out)
        return out