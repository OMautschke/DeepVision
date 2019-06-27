from Project.LoadData import LoadData


train_path = 'bdd100k/images/10k/'


def main():
    load_data = LoadData()
    test_loader, train_loader, val_loader = load_data.load_data(train_path)

    for batch_idx, (data, target) in enumerate(val_loader):
        print(target[0:16])
        pass


if __name__ == "__main__":
    main()
