import os
import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, MNIST

class CachedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_name, root, train=True, cache_dir='./cache'):
        self.cache_dir = os.path.join(cache_dir, dataset_name)
        os.makedirs(self.cache_dir, exist_ok=True)

        # select the dataset
        if dataset_name == "CIFAR10":
            self.dataset = CIFAR10(root=root, train=train, download=True)
        elif dataset_name == "CIFAR100":
            self.dataset = CIFAR100(root=root, train=train, download=True)
        elif dataset_name == "MNIST":
            self.dataset = MNIST(root=root, train=train, download=True)
        else:
            raise ValueError("Unsupported dataset")

        # define deterministic transformations (resize and normalize)
        if dataset_name in ["CIFAR10", "CIFAR100"]:
            self.transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        elif dataset_name == "MNIST":
            self.transform = transforms.Compose([
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # cache file path for the current image
        cache_path = os.path.join(self.cache_dir, f"{idx}.pt")

        # check if the preprocessed image is already cached
        if os.path.exists(cache_path):
            # Load the cached image
            image = torch.load(cache_path)
        else:
            # load the original image and apply deterministic transformations
            image, _ = self.dataset[idx]
            image = self.transform(image)

            # save the transformed image to the cache
            torch.save(image, cache_path)

        # get the label
        _, label = self.dataset[idx]
        return image, label
