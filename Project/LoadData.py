import os
import torch
import torchvision
import torchvision.transforms as transforms


class LoadData(object):

    def __init__(self):
        self.transform = transforms.Compose(
            [
             transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )

    def load_data(self, path):
        dataset = torchvision.datasets.ImageFolder(
            root=path,
            transform=self.transform
        )

        dirs = os.listdir(path)
        _, _, files = next(os.walk(path + dirs[0]))
        test_loader = torch.utils.data.DataLoader(
            dataset.imgs[0:len(files)],
            batch_size=1,   # Default 1
            shuffle=False
        )

        offset = len(files)
        _, _, files = next(os.walk(path + dirs[1]))
        train_loader = torch.utils.data.DataLoader(
            dataset.imgs[offset:(len(files) + offset)],
            batch_size=1,   # Default 1
            shuffle=True
        )

        offset = len(files)
        _, _, files = next(os.walk(path + dirs[2]))
        val_loader = torch.utils.data.DataLoader(
            dataset.imgs[offset:(len(files) + offset)],
            batch_size=1,   # Default 1
            shuffle=False
        )

        return test_loader, train_loader, val_loader
