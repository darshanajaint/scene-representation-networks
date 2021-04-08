from torch.nn import Module, Linear, Sigmoid
from torchvision.models import mobilenet_v2


class MobileNetModel(Module):
    def __init__(self):
        super(MobileNetModel, self).__init__()
        self.mobile = mobilenet_v2(pretrained=True)

        for param in self.mobile.parameters():
            param.requires_grad = False

        print(self.mobile)
        self.linear = Linear(1000, 1)
        # self.activation = Sigmoid()

    def forward(self, image):

        output = self.mobile(image)
        output = self.linear(output)
        # output = self.activation(output)

        return output
