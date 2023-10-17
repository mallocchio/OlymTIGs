# usage: python MNISTModel1.py - train the model
from keras.datasets import mnist
from keras.layers import Convolution2D, MaxPooling2D, Input, Dense, Activation, Flatten
from keras.models import Model
from keras.utils import to_categorical
from keras import backend as K
from tensorflow.keras.layers import Input
import torch

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def TF_LeNet1(input_tensor=None, train=False):
    nb_classes = 10
    # convolution kernel size
    kernel_size = (5, 5)

    if train:
        batch_size = 256
        nb_epoch = 10

        # input image dimensions
        img_rows, img_cols = 28, 28

        # the data, shuffled and split between train and test sets
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255

        # convert class vectors to binary class matrices
        y_train = to_categorical(y_train, nb_classes)
        y_test = to_categorical(y_test, nb_classes)

        input_tensor = Input(shape=input_shape)
    elif input_tensor is None:
        print('you have to proved input_tensor when testing')
        exit()

    # block1
    # print("in Model1 input_tensor = ",input_tensor)
    # default stride for conv2d is (1,1)
    x = Convolution2D(4, kernel_size, activation='relu', padding='same', name='block1_conv1')(input_tensor)
    # print("in Model1 x = ", x)
    x = MaxPooling2D(pool_size=(2, 2), name='block1_pool1')(x)
    #print("TF first block")
    #import tensorflow as tf
    #an_array = x.eval(session=tf.compat.v1.Session())

    # block2
    x = Convolution2D(12, kernel_size, activation='relu', padding='same', name='block2_conv1')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='block2_pool1')(x)

    x = Flatten(name='flatten')(x)
    x = Dense(nb_classes, name='before_softmax')(x)
    x = Activation('softmax', name='predictions')(x)

    model = Model(input_tensor, x)

    if train:
        # compiling
        model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

        # trainig
        model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=batch_size, epochs=nb_epoch, verbose=1)
        # save model
        model.save_weights('./Models/lenet1.h5')
        score = model.evaluate(x_test, y_test, verbose=0)
        print('\n')
        print('Overall Test score:', score[0])
        print('Overall Test accuracy:', score[1])
    else:
        model.load_weights('./Models/lenet1.h5')
        print('model loaded')

    return model


# pt model definition
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


def convert_tf_to_torch():
    # input image dimensions
    img_rows, img_cols = 28, 28
    img_dim = img_rows * img_cols
    input_shape = (img_rows, img_cols, 1)

    # load tf weights
    input_tensor = Input(shape=input_shape)
    model1 = TF_LeNet1(input_tensor=input_tensor)
    tf_weights = model1.get_weights()

    # load pt state_dict
    net = Torch_LeNet1()
    sd = net.state_dict()


    # copy tf weights to pt
    def translate_convw(weights, index):
        convw = weights[index]
        convw = np.transpose(convw, (3, 2, 0, 1))
        convw = torch.from_numpy(convw)
        return convw


    def translate_bias(weights, index):
        convb = weights[index]
        convb = torch.from_numpy(convb)
        return convb


    sd['conv1.weight'] = translate_convw(tf_weights, 0)
    sd['conv1.bias'] = translate_bias(tf_weights, 1)
    sd['conv2.weight'] = translate_convw(tf_weights, 2)
    sd['conv2.bias'] = translate_bias(tf_weights, 3)

    out_w = tf_weights[4]
    out_w = np.transpose(out_w)
    out_w = torch.from_numpy(out_w)
    sd['out.weight'] = out_w

    sd['out.bias'] = translate_bias(tf_weights, 5)

    torch.save(sd, "./Models/lenet1.pt")


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
