from Project.LoadData import LoadData
from Project.FeatureExtractor import FeatureExtractor
import torch
from Project.RPN import RPN
import matplotlib.pyplot as plt
from torch.autograd import Variable

test_path = 'bdd100k/images/10k/test/'
train_path = 'bdd100k/images/10k/train/'
val_path = 'bdd100k/images/10k/val/'

create_feature_map = True

activation = {}


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


def feature_map():
    load_data = LoadData()
    test_loader, train_loader, val_loader = load_data.load_data(test_path, train_path, val_path)

    if create_feature_map:
        feature_extractor = FeatureExtractor(train_loader)
        torch.save(feature_extractor.state_dict(), "./FeatureMap.pt")

    model = FeatureExtractor(train_loader)
    model.load_state_dict(torch.load("./FeatureMap.pt"))
    model.eval()

    rpn = RPN(512, 512, model.state_dict())

    #model.conv_trans2.register_forward_hook(get_activation("conv_trans2"))
    for n, p in model.named_parameters():
        #print(n, p.shape)
        pass


def load_my_state_dict(self, state_dict):

    own_state = self.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            continue
        if isinstance(param, torch.nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        own_state[name].copy_(param)


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
