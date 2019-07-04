import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureExtractor(nn.Module):

    def __init__(self, train_loader):
        super(FeatureExtractor, self).__init__()
        self.train_loader = train_loader

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.conv_trans1 = nn.ConvTranspose2d(16, 6, kernel_size=5, stride=1, padding=0)
        self.conv_trans2 = nn.ConvTranspose2d(6, 3, kernel_size=5, stride=1, padding=0)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)

        #out = out.reshape((1, 16 ))

        out = F.relu(self.conv_trans1(out))
        out = self.conv_trans2(out)
        return out

    def train(self):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        for epoch in range(1):
            for batch_idx, (data, _) in enumerate(self.train_loader):
                optimizer.zero_grad()
                output = self(data)

                loss = criterion(output, data)
                loss.backward()
                optimizer.step()

                print('Epoch {}, Batch idx {}, loss {}'.format(
                    epoch, batch_idx, loss.item()))
