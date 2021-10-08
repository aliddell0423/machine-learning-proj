import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from torch import nn
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, Lambda
from torch.utils.data import DataLoader

import PIL.ImageOps
import requests
from PIL import Image

transform  = transforms.Compose([transforms.Resize((28,28)),
                                 transforms.ToTensor()
                                ])

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=False,
    transform=transform
)


training_data, validation_data = torch.utils.data.random_split(training_data, [50000, 10000])

class Model(nn.Module):
    def __init__(self, input_size, h1, h2, output_size):
        super().__init__()
        self.layer_1 = nn.Linear(input_size, h1)
        self.layer_2 = nn.Linear(h1, h2)
        self.layer_3 = nn.Linear(h2, output_size)
    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = self.layer_3(x)
        return x

model = Model(784, 300, 100, 10)

training_loader = torch.utils.data.DataLoader(dataset=training_data,
                                              batch_size=100,
                                              shuffle=True)

# We don't need to shuffle validation data
validation_loader = torch.utils.data.DataLoader(dataset=validation_data,
                                                batch_size=100,
                                                shuffle=False)

# Cross Entropy Loss with Adam Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 3

for e in range(epochs):

    train_corrects = 0.0
    train_batch_loss = 0.0
    train_epoch_loss = 0.0
    val_corrects = 0.0
    val_epoch_loss = 0.0

    # loop through 60000 samples 100 at a time
    for batch_idx, data in enumerate(training_loader, start=1):
        # convert (100, 1, 28 , 28) to (100, 748)
        # data[0] is the array containing 100(batch_size) number of 1*28*28
        # data[1] is the array containing 100 number of lables
        inputs = data[0].view(data[0].shape[0], -1)
        labels = data[1]
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Return the index of the highest possibility
        # which are the predicted labels
        _, preds = torch.max(outputs, 1)
        train_batch_loss += loss.item()

        # sum up all the correct prediction
        # since (preds==labels).sum() is a tensor
        # we use item() to extract the number
        train_corrects += (preds == labels).sum().item()

        # print training loss every 100 mini-batch
        # train_batch_loss is the average loss for 100 mini-batch
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                e + 1,
                batch_idx * len(data[0]),
                len(training_loader.dataset),
                100. * batch_idx * len(data[0]) / len(training_loader.dataset),
                train_batch_loss / 100))
            # accumulate loss for the epoch
            train_epoch_loss += train_batch_loss
            # reset the loss for every mini-batch
            train_batch_loss = 0.0
    else:
        # torch.no_grad deactivate the auograd engine,
        # reduce memory usage and speed up computations
        with torch.no_grad():
            for val_data in validation_loader:
                val_inputs = val_data[0].view(val_data[0].shape[0], -1)
                val_labels = val_data[1]
                val_outputs = model(val_inputs)
                val_loss = criterion(val_outputs, val_labels)

                _, val_preds = torch.max(val_outputs, 1)
                val_epoch_loss += val_loss.item()
                val_corrects += (val_preds == val_labels).sum().item()

        # print result for every epoch
        train_accuracy = 100. * train_corrects / len(training_loader.dataset)
        # here batch_idx is the total number of mini-batch = 600
        train_epoch_loss /= batch_idx

        print('epoch :', (e + 1))
        print('Train set: Accuracy: {}/{} ({:.0f}%), Average Loss: {:.6f}'.format(
            train_corrects, len(training_loader.dataset),
            train_accuracy, train_epoch_loss))

        val_accuracy = 100. * val_corrects / len(validation_loader.dataset)
        val_epoch_loss /= batch_idx

        print('Validation set: Accuracy: {}/{} ({:.0f}%), Average Loss: {:.6f}'.format(
            val_corrects, len(validation_loader.dataset),
            val_accuracy, val_epoch_loss))