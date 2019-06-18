import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(CNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(6, 25, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 25, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc1 = nn.Linear("?", 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)

        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))

        return out


"""
https://www.learnopencv.com/number-of-parameters-and-tensor-sizes-in-convolutional-neural-network/
Output Size:
(64 x 64 x 1): Activation size = 64 * 64 * 1 = 4096

Layer_1: 64 - 5 + 1 = 60 x 60 x 6
MaxPool: (60 - 2) / 2 + 1 = 30 x 30 x 6

Layer_2: 30 - 5 + 1 = 26 x 26 x 16
MaxPool: (30 - 2) / 2 + 1 = 15 x 15 x 16

fc1: 120 x 1
fc2:  84 x 1
fc3:  20 x 1

Number of Parameters:
Layer1:
Weights = 5^2 x 1 x 6 = 150
Biases = 6
Parameters = 150 + 6 = 156

Layer2:
Weights = 5^2 x 1 x 16 = 400
Biases = 16
Parameters = 400 + 16 = 416


fc1 connected to a Conv Layer:
Weights: 15^2 x 16 x 120 = 432000
Biases:  120
Parameters = 432120


fc2 connected to the previous fc layer:
Weights: 120 x 84 = 10080
Biases:  84
Parameters: 10164

fc3 connected to the previous fc layer:
Weights: 84 x 10 = 840
Biases:  10
Parameters: 850

Total Parameters: 156 + 416 + 432120 + 10164 + 850 = 443706
"""
