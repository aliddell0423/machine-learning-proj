import torch
import matplotlib.pyplot as plt
import numpy
import torch.nn.functional as F
from torch import nn
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, Lambda
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

transform = transforms.Compose([transforms.Resize((28, 28)),transforms.ToTensor()])

full_data_dict = {}
for data_name in ["FashionMNIST"]: # For Extra Credit add another data set.
    data_loader = getattr(datasets, data_name)
    full_data_dict[data_name] = data_loader(
	  root="data",
	  train=False,
	  download=True,
	  transform=ToTensor()
    )

training_data = datasets.FashionMNIST(
 root="data",
 train=True,
 download=False,
 transform=transform
)

throw_away, whole = torch.utils.data.random_split(training_data, [59400, 600])

validation_data, training_data = torch.utils.data.random_split(whole, [100, 500])


# Create our model
class LeNet(nn.Module):

     def __init__(self):
         super().__init__()
         self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
         self.conv2 = nn.Conv2d(6, 16, 5)
         self.fc1 = nn.Linear(4 * 4 * 16, 120)
         self.fc2 = nn.Linear(120, 84)
         self.output = nn.Linear(84, 10)

     def forward(self, x):
         x = F.relu(self.conv1(x))
         x = F.max_pool2d(x, 2, 2)
         x = F.relu(self.conv2(x))
         x = F.max_pool2d(x, 2, 2)
         x = x.view(-1, 4 * 4 * 16)
         x = F.relu(self.fc1(x))
         x = F.relu(self.fc2(x))
         x = self.output(x)
         return x

class Connected(nn.Module):
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

# Define our models
lenet = LeNet()
connected = Connected(784, 300, 100, 10)

# DataLoader
training_loader = torch.utils.data.DataLoader(dataset=training_data,
 batch_size=8,
 shuffle=True)

# We don't need to shuffle validation data
validation_loader = torch.utils.data.DataLoader(dataset=validation_data,
 batch_size=8,
 shuffle=False)

# Cross Entropy Loss with Adam Optimizer
criterion = nn.CrossEntropyLoss()

epochs = 600

for model in lenet, connected:
     optimizer = torch.optim.Adam(model.parameters(), lr=0.04)
     writer_dict = {}
     for s in "subtrain", 'validation':
        writer_dict[s] = SummaryWriter("runs/" + s)
     for e in range(epochs):

         train_corrects = 0.0
         train_batch_loss = 0.0
         train_epoch_loss = 0.0
         val_corrects = 0.0
         val_epoch_loss = 0.0

         # loop through 60000 samples 100 at a time
         for batch_idx, data in enumerate(training_loader, start=1):
             if model is lenet:
                 inputs = data[0]
             else:
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
             if batch_idx % 8 == 0:
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
         elif :
             # torch.no_grad deactivate the auograd engine,
             # reduce memory usage and speed up computations
             with torch.no_grad():
                 for val_data in validation_loader:
                     if model is lenet:
                         val_inputs = val_data[0]
                     else:
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

             writer_dict["subtrain"].add_scalar(str(model) + '/loss', train_epoch_loss, e + 1)

             print('epoch :', (e + 1))
             print('Train set: Accuracy: {}/{} ({:.0f}%), Average Loss: {:.6f}'.format(
                 train_corrects, len(training_loader.dataset),
                 train_accuracy, train_epoch_loss))

             val_accuracy = 100. * val_corrects / len(validation_loader.dataset)
             val_epoch_loss /= batch_idx

             writer_dict["validation"].add_scalar(str(model) + '/loss', val_epoch_loss, e + 1)

             print('Validation set: Accuracy: {}/{} ({:.0f}%), Average Loss: {:.6f}'.format(
                 val_corrects, len(validation_loader.dataset),
                 val_accuracy, val_epoch_loss))
