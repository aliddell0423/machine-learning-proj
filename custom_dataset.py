import torch
import torch.nn.functional as F
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader


def TestErrorOneSplit(data_name, splitNum, model):
    results = {}
    train_set, test_set = getTrainTestData(data_name, splitNum)
    testloader = torch.utils.data.DataLoader(dataset=test_set, batch_size=100, shuffle=False)
    correct, total = 0, 0
    if model == "featureless":
        predicted = getMostFrequentLabel(train_set)
    else:
        subtrain_set, validation_set = splitData(train_set)
        subtrain_net = newModel(model)
        subtrain_result = learn(subtrain_net, subtrain_set, MAX_EPOCHS, validation=validation_set)
        print("DATA=%s model=%s splitNum=%d selectedEpochs=%d" % (data_name, model, splitNum, subtrain_result))
        train_net = newModel(model)
        learn(train_net, train_set, subtrain_result)

    with torch.no_grad():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Iterate over the test data and generate predictions
        for i, data in enumerate(testloader, start=1):
            # Get inputs
            if model == "convolutional":
                inputs = data[0]
            elif model == "fullyConnected":
                inputs = data[0].view(data[0].shape[0], -1)
            targets = data[1]

            if model != "featureless":
                inputs, targets = inputs.to(device), targets.to(device)

            if model != "featureless":
                outputs = train_net(inputs)
                _, predicted = torch.max(outputs.data, 1)


            # Set total and correct
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        # Print accuracy
        print('Error Rate for fold %d: %d %%' % (splitNum, 100 - (100.0 * correct / total)))
        print('--------------------------------')
        results[splitNum] = 100.0 * (correct / total)


def learn(model, train_set, epochs, validation=None):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    val_min = 10000

    min_epochs = 0

    model.to(device)

    if validation is not None:
        validationloader = torch.utils.data.DataLoader(dataset=validation, batch_size=100, shuffle=False)

    trainloader = torch.utils.data.DataLoader(dataset=train_set, batch_size=100, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    loss_function = nn.CrossEntropyLoss()

    # Run the training loop for defined number of epochs
    for e in range(epochs):

        train_corrects = 0.0
        train_batch_loss = 0.0
        train_epoch_loss = 0.0
        val_corrects = 0.0
        val_epoch_loss = 0.0

        # loop through 60000 samples 100 at a time
        for batch_idx, data in enumerate(trainloader, start=1):
            if isinstance(model, LeNet):
                inputs = data[0]
            else:
                inputs = data[0].view(data[0].shape[0], -1)
            labels = data[1]
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_function(outputs, labels)

            if validation is not None:
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
                    len(trainloader.dataset),
                    100. * batch_idx * len(data[0]) / len(trainloader.dataset),
                    train_batch_loss / 100))
                # accumulate loss for the epoch
                train_epoch_loss += train_batch_loss
                # reset the loss for every mini-batch
                train_batch_loss = 0.0
            elif validation is not None:
                # torch.no_grad deactivate the auograd engine,
                # reduce memory usage and speed up computations
                with torch.no_grad():
                    for val_data in validationloader:
                        if isinstance(model, LeNet):
                            val_inputs = val_data[0]
                        else:
                            val_inputs = val_data[0].view(val_data[0].shape[0], -1)
                        val_labels = val_data[1]
                        val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                        val_outputs = model(val_inputs)
                        val_loss = loss_function(val_outputs, val_labels)

                    _, val_preds = torch.max(val_outputs, 1)
                    val_epoch_loss += val_loss.item()
                    val_corrects += (val_preds == val_labels).sum().item()

            # print result for every epoch
            train_accuracy = 100. * train_corrects / len(trainloader.dataset)
            # here batch_idx is the total number of mini-batch = 600
            train_epoch_loss /= batch_idx

            print('epoch :', (e + 1))
            print('Train set: Accuracy: {}/{} ({:.0f}%), Average Loss: {:.6f}'.format(
                train_corrects, len(trainloader.dataset),
                train_accuracy, train_epoch_loss))

            if validation is not None:
                val_accuracy = 100. * val_corrects / len(validationloader.dataset)
                val_epoch_loss /= batch_idx

                if val_min > val_epoch_loss:
                    val_min = val_epoch_loss
                    min_epochs = e + 1

                print('Validation set: Accuracy: {}/{} ({:.0f}%), Average Loss: {:.6f}'.format(
                    val_corrects, len(validationloader.dataset),
                    val_accuracy, val_epoch_loss))

    return min_epochs


def splitData(train_set):
    subtrain_set, validation_set = torch.utils.data.random_split(train_set, [5556, 1111],
                                            generator=torch.Generator().manual_seed(splitNum))
    return subtrain_set, validation_set


def getTrainTestData( data_name, splitNum ):
    data_loader = getattr(datasets, data_name)
    dataset = data_loader(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )

    train_set, test_set = torch.utils.data.random_split(dataset, [6667, 3333],
                                            generator=torch.Generator().manual_seed(splitNum))
    return train_set, test_set


def getMostFrequentLabel(train_set):
    temp = torch.utils.data.DataLoader(dataset=train_set,
            shuffle=False)

    label_dict = {}

    for batch_idx, data in enumerate(temp, start=1):
        if data[1].item() in label_dict:
            label_dict[data[1].item()] += 1
        else:
            label_dict[data[1].item()] = 0

    max_label_num = 0
    max_label = None

    for label in label_dict:
        if label_dict[label] > max_label_num:
            max_label_num = label_dict[label]
            max_label = label

    for batch_idx, data in enumerate(temp, start=1):
        if data[1].item() == max_label:
            return data[1]





def newModel(model):
    if model is 'convolutional':
        return LeNet()
    else:
        return Connected(784, 300, 100, 10)


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


if __name__ == "__main__":
    MAX_EPOCHS = 50
    NUM_SPLITS = 3

    for data_name in ["FashionMNIST"]:
        for splitNum in range(NUM_SPLITS):
            for model in "featureless", "convolutional", "fullyConnected":
                TestErrorOneSplit(data_name, splitNum, model)
