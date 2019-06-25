'''MobileNet in PyTorch.
See the paper "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision


class Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out


class MobileNet(nn.Module):
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]

    def __init__(self, num_classes=10):
        super(MobileNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.linear = nn.Linear(1024, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def train(self):
        trainloader, _ = self.load_data()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(mobileNet.parameters(), lr=0.001, momentum=0.9)

        for epoch in range(5):
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data

                optimizer.zero_grad()
                outputs = mobileNet(inputs)

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
                outputs = mobileNet(images)
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
    mobileNet = MobileNet()
    mobileNet.train()
    mobileNet.eval()
