import torch
from torch import nn

class PT_MLP(nn.Module):
    def __init__(self, size_hidden=100, size_out=10):
        super().__init__()

        self.fc0 = nn.Linear(28*28, size_hidden)
        self.fc1 = nn.Linear(size_hidden, size_hidden)
        self.fc2 = nn.Linear(size_hidden, size_hidden)
        self.fc3 = nn.Linear(size_hidden, size_hidden)
        self.fc4 = nn.Linear(size_hidden, size_out)

        self.relu = nn.ReLU()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y):
        z0 = self.fc0(x)
        z1 = self.relu(z0)
        z2 = self.fc1(z1)
        z3 = self.relu(z2)
        z4 = self.fc2(z3)
        z5 = self.relu(z4)
        z6 = self.fc3(z5)
        z7 = self.relu(z6)
        z8 = self.fc4(z7)
        z9 = self.loss(z8, y)
        endpoints = locals()
        names = ["z{}".format(i) for i in range(10)]
        endpoints = dict((k, endpoints[k]) for k in names)
        return endpoints
