from torch.utils.data import DataLoader
import torchvision

def get_batches(train, batch_size, shuffle = True):
    img_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = torchvision.datasets.MNIST('./data', train = train, transform = img_transform, download = True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
