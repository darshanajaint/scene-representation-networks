import torch
from MobileNetModel import MobileNetModel


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MobileNetModel()