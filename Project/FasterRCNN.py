import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


class FasterRCNN(nn.Module):

    def __init__(self):
        super(FasterRCNN, self).__init__()
