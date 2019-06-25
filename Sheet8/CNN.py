import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)

        out = out.view(-1, 16 * 5 * 5)

        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)

        return out

    def train(self):
        trainloader, _ = self.load_data()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(cnn.parameters(), lr=0.001, momentum=0.9)

        for epoch in range(10):
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data

                optimizer.zero_grad()
                outputs = cnn(inputs)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

    def eval(self):
        _, testloader = self.load_data()
        total_correct = 0
        total_images = 0

        with torch.no_grad():
            for data in testloader:
                images, labels = data
                outputs = cnn(images)
                _, predicted = torch.max(outputs.data, 1)

                total_images += labels.size(0)
                total_correct += (predicted == labels).sum().item()

        model_accuracy = total_correct / total_images * 100
        print('Model accuracy on {0} test images: {1:.2f}%'.format(total_images, model_accuracy))

    def load_data(self):
        transform = transforms.Compose(
            [
             transforms.RandomHorizontalFlip(p=0.5),
             transforms.RandomCrop(32, padding=4),
             transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
             ])

        trainset = torchvision.datasets.CIFAR10(root='./Data',
                                                train=True,
                                                download=True,
                                                transform=transform)

        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=16,
                                                  shuffle=True)

        testset = torchvision.datasets.CIFAR10(root='./Data',
                                               train=False,
                                               download=True,
                                               transform=transform)

        testloader = torch.utils.data.DataLoader(testset,
                                                 batch_size=16,
                                                 shuffle=False)

        return trainloader, testloader


if __name__ == "__main__":
    cnn = CNN()
    cnn.train()
    cnn.eval()

