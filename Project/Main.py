from Project.LoadData import LoadData
from Project.FeatureExtractor import FeatureExtractor
from Project.RPN import RPN


test_path = 'bdd100k/images/10k/test/'
train_path = 'bdd100k/images/10k/train/'
val_path = 'bdd100k/images/10k/val/'


def main():
    load_data = LoadData()
    test_loader, train_loader, val_loader = load_data.load_data(test_path, train_path, val_path)

    for batch_idx, (data, target) in enumerate(test_loader):
        #print(data)
        pass

    feature_extractor = FeatureExtractor(train_loader)
    feature_extractor.train()


if __name__ == "__main__":
    main()
