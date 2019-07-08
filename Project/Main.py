from Project.LoadData import LoadData
from Project.FeatureExtractor import FeatureExtractor
import torch
from Project.RPN import RPN
import matplotlib.pyplot as plt


test_path = 'bdd100k/images/10k/test/'
train_path = 'bdd100k/images/10k/train/'
val_path = 'bdd100k/images/10k/val/'

create_feature_map = True


def feature_map():
    load_data = LoadData()
    test_loader, train_loader, val_loader = load_data.load_data(test_path, train_path, val_path)

    if create_feature_map:
        feature_extractor = FeatureExtractor(train_loader)
        torch.save(feature_extractor.state_dict(), "./FeatureMap.pt")

    model = FeatureExtractor(train_loader)
    model.load_state_dict(torch.load("./FeatureMap.pt"))
    model.eval()


def main():
    feature_map()

    """
    for batch_idx, (data, target) in enumerate(test_loader):
        #print(data)
        pass
    
    
    from torchvision.utils import make_grid

    kernels = feature_extractor.layer1[0].weight.detach().clone()
    kernels = kernels - kernels.min()
    kernels = kernels / kernels.max()
    img = make_grid(kernels)
    plt.imshow(img.permute(1, 2, 0))
    plt.show()
    """


if __name__ == "__main__":
    main()
