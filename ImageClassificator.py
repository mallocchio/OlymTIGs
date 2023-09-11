import numpy as np 
import torch 
import torchvision 
import matplotlib.pyplot as plt 
from time import time 
from torchvision import datasets, transforms 
from torch import nn, optim
import pandas as pd

# questa funzione permette di modificare le immagini in modo che siano tutte delle stesse dimensioni e caratteristiche 
transform = transforms.Compose([transforms.ToTensor(), 
                              transforms.Normalize((0.5,), (0.5,)), 
                              ])

# scarico i set di dati
trainset = datasets.MNIST('PATH_TO_STORE_TRAINSET', download= True , train= True , transform=transform)

valset = datasets.MNIST('PATH_TO_STORE_TESTSET', download= True , train= False , transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle= True )

valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle= True )

dataiter = iter(trainloader)
images, labels = next(dataiter)

input_size = 784
hidden_sizes = [128, 64]
output_size = 10

model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.LogSoftmax(dim=1))
print(model)

criterion = nn.NLLLoss()
images, labels = next(iter(trainloader))
images = images.view(images.shape[0], -1)
print(images[0].shape)

logps = model(images) #log probabilities
loss = criterion(logps, labels) #calculate the NLL loss

print('Before backward pass: \n', model[0].weight.grad)
loss.backward()
print('After backward pass: \n', model[0].weight.grad)

criterion = nn.NLLLoss()
images, labels = next(iter(trainloader))
images = images.view(images.shape[0], -1)

logps = model(images) #log probabilities
loss = criterion(logps, labels) #calculate the NLL loss


# iteri il processso per n volte in modo che i valori si calibrino dando un riscontro sempre più preciso minimizzando gli errori
optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
time0 = time()
epochs = 15
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        # Flatten MNIST images into a 784 long vector
        images = images.view(images.shape[0], -1)
    
        # Training pass
        optimizer.zero_grad()
        
        output = model(images)
        loss = criterion(output, labels)
        
        #This is where the model learns by backpropagating
        loss.backward()
        
        #And optimizes its weights here
        optimizer.step()
        
        running_loss += loss.item()
    else:
        print("Epoch {} - Training loss: {}".format(e, running_loss/len(trainloader)))
print("\nTraining Time (in minutes) =",(time()-time0)/60)



images, labels = next(iter(valloader))

img = images[0].view(1, 784)
with torch.no_grad():
    logps = model(img)

ps = torch.exp(logps)
probab = list(ps.numpy()[0])
print("Predicted Digit =", probab.index(max(probab)))

correct_count, all_count = 0, 0
for images,labels in valloader:
  for i in range(len(labels)):
    img = images[i].view(1, 784)
    with torch.no_grad():
        logps = model(img)

    
    ps = torch.exp(logps)
    probab = list(ps.numpy()[0])
    pred_label = probab.index(max(probab))
    true_label = labels.numpy()[i]
    if(true_label == pred_label):
      correct_count += 1
    all_count += 1

print("Number Of Images Tested =", all_count)
print("\nModel Accuracy =", (correct_count/all_count))

torch.save(model, './model.pt') 