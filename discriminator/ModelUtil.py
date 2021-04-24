import torch
import math
import numpy as np

from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from discriminator.ImageDataset import ImageDataset
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
    def __init__(self, model, transform, device, batch_size=128,
                 val_proportion=0.1):
        self.model = model
        self.transform = transform
        self.batch_size = batch_size
        self.device = device
        self.val_proportion = val_proportion
        self.criterion = BCEWithLogitsLoss()
        self.optimizer = Adam(self.model.parameters(), lr=0.0001)

        self.model = self.model.to(device)
        self.criterion = self.criterion.to(device)

    def _get_data_loader(self, reals, fakes, shuffle=True):
        # reals = torch.stack(reals)
        # fakes = torch.stack(fakes)
        data_set = ImageDataset(reals, fakes, self.transform)
        data_loader = DataLoader(data_set, self.batch_size, shuffle=shuffle)
        return data_loader

    def set_train(self):
        for param in self.model.linear.parameters():
            param.requires_grad = True

    def get_gan_loss(self, real, fake):
        m = float(len(real))
        loss = 1 / m * torch.sum(torch.log2(real) + torch.log2(1 - fake))
        return loss

    def _main_loop(self, data_loader, train):
        loss_epoch = 0
        num_batches = 0
        real_preds = []
        fake_preds = []

        for real, fake in data_loader:
            real = real.to(self.device)
            fake = fake.to(self.device)

            real_output = self.model(real)
            fake_output = self.model(fake)
            loss = self.get_gan_loss(real_output, fake_output)

            if train:
                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)
                self.optimizer.step()

            real_preds.extend((real_output.detach().cpu().numpy() >=
                               0.5).astype('float'))
            fake_preds.extend((fake_output.detach().cpu().numpy() >=
                               0.5).astype('float'))
            loss_epoch += loss.item()
            num_batches += 1

        loss_per_batch = loss_epoch / num_batches
        total_accuracy = accuracy_score(
            [1] * len(real_preds) + [0] * len(fake_preds),
            real_preds + fake_preds
        )
        real_accuracy = accuracy_score([1] * len(real_preds), real_preds)
        fake_accuracy = accuracy_score([0] * len(fake_preds), fake_preds)
        return loss_per_batch, total_accuracy, real_accuracy, fake_accuracy

    def train(self, reals, fakes):
        val_size = math.floor(len(reals) * self.val_proportion)
        train = self._get_data_loader(reals[:-val_size], fakes[:-val_size])
        val = self._get_data_loader(reals[-val_size:], fakes[-val_size:])

        self.set_train()
        self.model.train()

        loss, total, real_acc, fake_acc = self._main_loop(train, train=True)
        train_loss = loss
        train_total_acc = total
        train_real_acc = real_acc
        train_fake_acc = fake_acc

        self.model.eval()
        with torch.no_grad():
            loss, total, real_acc, fake_acc = self._main_loop(val, train=False)
            val_loss = total
            val_total_acc = total
            val_real_acc = real_acc
            val_fake_acc = fake_acc

        results = {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_total_accuracy': train_total_acc,
            'val_total_accuracy': val_total_acc,
            'train_real_accuracy': train_real_acc,
            'val_real_accuracy': val_real_acc,
            'train_fake_accuracy': train_fake_acc,
            'val_fake_accuracy': val_fake_acc
        }
        return results

    def predict(self, data):
        data_loader = self._get_data_loader(data, data, shuffle=False)

        self.model.eval()

        proba_list = []
        pred_list = []
        # with torch.no_grad():
        for image, _ in data_loader:
            image = image.to(self.device)

            output = self.model(image)
            # output = output.cpu().numpy()

            proba_list += list(output)

            output = (output >= 0.5).float()
            pred_list += list(output)

        return proba_list, pred_list

    def predict_proba(self, real, fake):
        self.model.eval()
        real_pred = self.model(real)
        fake_pred = self.model(fake)
        return [real_pred, fake_pred]
