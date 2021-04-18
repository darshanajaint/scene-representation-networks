import torch
import numpy as np

from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from ImageDataset import ImageDataset
from sklearn.metrics import accuracy_score, balanced_accuracy_score


# DONE:
#   - implement 3 model types - MobileNet v2, ResNet, GoogLeNet
#   - set num epochs = 10
#   - determine data set and data loader for images
#   - function to predict on images
#   - function to take image set and create train/val set out of them
#   - transform function - size of 224 x 224
#   - look at classification and balanced accuracies?
#   - basic model training
#   - plot training and validation losses and accuracies
#   - check project proposal for discriminator evaluation metrics
#   - evaluation metrics - confusion matrices, precision, recall
class ModelUtil:
    def __init__(self, model, transform, device, batch_size=128, num_epochs=10):
        self.model = model
        self.transform = transform
        self.batch_size = batch_size
        self.device = device
        self.criterion = BCEWithLogitsLoss()
        self.optimizer = Adam(self.model.parameters(), lr=0.0001)

        self.model = self.model.to(device)
        self.criterion = self.criterion.to(device)
        self.num_epochs = num_epochs

    def _get_data_loader(self, data, labels):
        data = torch.Tensor(data)
        labels = torch.Tensor(labels)
        data_set = ImageDataset(data, labels, self.transform)
        data_loader = DataLoader(data_set, self.batch_size, shuffle=True)
        return data_loader

    def _main_loop(self, data_loader, train):
        loss_epoch = 0
        num_loops = 0
        preds = []
        labels = []

        for batch in data_loader:
            image = batch.image.to(self.device)
            label = batch.label.to(self.device)

            output = self.model(image)
            loss = self.criterion(output, label)

            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            preds += list((output.cpu().numpy() >= 0.5).float())
            labels += list(label.cpu().numpy())

            loss_epoch += loss.item()
            num_loops += 1
        return loss_epoch / num_loops, accuracy_score(labels, preds), \
            balanced_accuracy_score(labels, preds)

    def train(self, train_data, train_labels, val_data, val_labels):
        train_data_loader = self._get_data_loader(train_data, train_labels)
        val_data_loader = self._get_data_loader(val_data, val_labels)

        train_loss = []
        val_loss = []
        train_acc = []
        val_acc = []
        for epoch in range(self.num_epochs):
            self.model.train()

            loss, accuracy, balanced_accuracy = self._main_loop(
                train_data_loader, train=True)
            train_loss.append(loss)
            train_acc.append([accuracy, balanced_accuracy])

            self.model.eval()
            with torch.no_grad():
                loss, accuracy, balanced_accuracy = self._main_loop(
                    val_data_loader, train=False)
                val_loss.append(loss)
                val_acc.append([accuracy, balanced_accuracy])

        train_acc = np.asarray(train_acc)
        val_acc = np.asarray(val_acc)
        return [train_acc, train_loss], [val_acc, val_loss]

    def predict(self, data, labels):
        data_loader = self._get_data_loader(data, labels)

        self.model.eval()

        output_list = []
        labels_list = []
        with torch.no_grad():
            for batch in data_loader:
                image = batch.image.to(self.device)
                label = batch.label.to(self.device)

                output = self.model(image)
                output = torch.sigmoid(output)
                output = output.cpu().numpy()
                output = (output >= 0.5).float()

                labels_list += list(label.cpu().numpy())
                output_list += list(output)

        return output_list, labels_list
