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
        print("Original: ", x.size())

        x = self.layer1(x)
        print("Layer 1: ", x.size())

        x = self.layer2(x)
        print("Layer 2: ", x.size())

        #x = x.view((1 * 16, x.size()[2], x.size()[3]))
        #print("Reshape: ", x.size())

        x = F.relu(self.conv_trans1(x))
        print("Conv1: ", x.size())

        x = self.conv_trans2(x)
        print("Conv2: ", x.size())
        print("\n\n")
        return x

    """
    def train(self):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        for epoch in range(1):
            for batch_idx, (data, _) in enumerate(self.train_loader):
                optimizer.zero_grad()
                output = self(data)
                data = torch.tensor(data, dtype=torch.long)

                loss = criterion(output, data)
                loss.backward()
                optimizer.step()

                print('Epoch {}, Batch idx {}, loss {}'.format(
                    epoch, batch_idx, loss.item()))
    """
