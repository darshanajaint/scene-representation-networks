from torch.nn import Module, Linear, Sigmoid
from torchvision.models import mobilenet_v2, resnet34, googlenet


class MobileNetModel(Module):
    def __init__(self):
        super(MobileNetModel, self).__init__()
        self.mobile = mobilenet_v2(pretrained=True)
        self.linear = Linear(1000, 1)
        self.activation = Sigmoid()
        for param in self.mobile.parameters():
            param.requires_grad = False

    def forward(self, image):
        output = self.mobile(image)
        output = self.linear(output)
        output = self.activation(output)
        return output


class ResNetModel(Module):
    def __init__(self):
        super(ResNetModel, self).__init__()
        self.res = resnet34(pretrained=True)
        self.linear = Linear(1000, 1)
        self.activation = Sigmoid()
        for param in self.res.parameters():
            param.requires_grad = False

    def forward(self, image):
        output = self.res(image)
        output = self.linear(output)
        output = self.activation(output)
        return output


class GoogLeNetModel(Module):
    def __init__(self):
        super(GoogLeNetModel, self).__init__()
        self.goog = googlenet(pretrained=True)
        self.linear = Linear(1000, 1)
        self.activation = Sigmoid()
        for param in self.goog.parameters():
            param.requires_grad = False

    def forward(self, image):
        output = self.goog(image)
        output = self.linear(output)
        output = self.activation(output)
        return output
