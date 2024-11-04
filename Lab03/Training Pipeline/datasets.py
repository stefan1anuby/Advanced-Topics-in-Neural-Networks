import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from cached_dataset import CachedDataset 

def get_dataset(dataset_name, batch_size, use_cache=False):
    if use_cache:
        train_data = CachedDataset(dataset_name=dataset_name, root='./data', train=True)
        test_data = CachedDataset(dataset_name=dataset_name, root='./data', train=False)
    else:
        if dataset_name == "MNIST":
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
            test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        elif dataset_name == "CIFAR10":
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
            test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        elif dataset_name == "CIFAR100":
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            train_data = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
            test_data = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
        else:
            raise ValueError("Dataset not supported!")

    # create DataLoaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
