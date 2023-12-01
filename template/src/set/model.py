import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class SubClass:
    pass

class MainClass(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        pass

    def forward(self, x):
        pass


def test():
    #define dummy input
    #create model instance
    #preds = model(x)
    #print debug
    # assert xxxx

if __name__ == "__main__":
    test()