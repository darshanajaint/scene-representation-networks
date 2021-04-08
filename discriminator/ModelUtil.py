import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from MobileNetModel import MobileNetModel


class ModelUtil:
    def __init__(self, batch_size, device):
        self.model = MobileNetModel()
        self.batch_size = batch_size
        self.device = device
        self.criterion = BCEWithLogitsLoss()
        self.optimizer = Adam(self.model.parameters(), lr=0.0001)

        self.model = self.model.to(device)
        self.criterion = self.criterion.to(device)

    def _get_data_loader(self, data, labels):
        data = torch.Tensor(data)
        labels = torch.Tensor(labels)
        data_set = TensorDataset(data, labels)
        data_loader = DataLoader(data_set, self.batch_size, shuffle=True)
        return data_loader

    def train(self, data, labels):
        data_loader = self._get_data_loader(data, labels)

        num_epochs = 5
        for epoch in range(num_epochs):
            self.model.train()
            for batch in data_loader:
                image = batch.image.to(self.device)
                label = batch.label.to(self.device)

                output = self.model(image)
                loss = self.criterion(output, label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def predict(self, data, labels):
        data_loader = self._get_data_loader(data, labels)

        self.model.eval()

        output_list = []
        with torch.no_grad():
            for batch in data_loader:
                image = batch.image.to(self.device)
                label = batch.label.to(self.device)

                output = self.model(image)
                output = torch.sigmoid(output)
                output = output.cpu().numpy()
                output = (output >= 0.5).float()

                output_list += output

        return output_list


