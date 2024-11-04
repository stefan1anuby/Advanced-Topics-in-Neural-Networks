import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import wandb

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="my-awesome-project"
)

basic_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor()
])

advanced_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomRotation(15),
    transforms.ToTensor()
])

def get_dataloader(transform):
    train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transforms.ToTensor())
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    return train_loader, test_loader

def train(model, train_loader, test_loader, optimizer, criterion, epochs=10):
    model.train()
    test_accuracy = 0.0
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / total
        train_accuracy = correct / total

        wandb.log({
            "Train Loss": train_loss,
            "Train Accuracy": train_accuracy,
            "Epoch": epoch
        })

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        test_accuracy = correct / total
        wandb.log({"Test Accuracy": test_accuracy})

    return test_accuracy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models = {
          "VGG16": torchvision.models.vgg16(num_classes=100),
          "ResNet18": torchvision.models.resnet18(num_classes=100),
          }
augmentations = {"basic": basic_transforms,
                 "advanced": advanced_transforms
                 }
optimizers = {"SGD": optim.SGD,
              "Adam": optim.Adam
              }
learning_rates = [0.01]
print(device)

import datetime
results = []

for model_name, model in models.items():
    for aug_name, aug_transform in augmentations.items():
        for opt_name, opt_class in optimizers.items():
            for lr in learning_rates:
                current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                print("Config: " + current_time)

                # create a unique project name with model, augmentation, optimizer, and date-time
                project_name = f"{model_name}_{aug_name}_{opt_name}_{current_time}"

                # start a new wandb run for each configuration
                wandb.init(project=project_name, config={
                    "Model": model_name,
                    "Augmentation": aug_name,
                    "Optimizer": opt_name,
                    "Learning Rate": lr
                })

                # load data with the chosen augmentation
                train_loader, test_loader = get_dataloader(aug_transform)

                # initialize the model and move it to the device
                model = model.to(device)

                # define the optimizer and loss criterion
                optimizer = opt_class(model.parameters(), lr=lr)
                criterion = nn.CrossEntropyLoss()

                # log the configuration
                wandb.config.update({
                    "Model": model_name,
                    "Augmentation": aug_name,
                    "Optimizer": opt_name,
                    "Learning Rate": lr
                }, allow_val_change=True)

                # train the model
                test_accuracy = train(model, train_loader, test_loader, optimizer, criterion, epochs=10)

                results.append({
                    "Model": model_name,
                    "Augmentation": aug_name,
                    "Optimizer": opt_name,
                    "Learning Rate": lr,
                    "Test Accuracy": test_accuracy
                })

                # reset the model for the next run
                model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)

# Display the results in a table
import pandas as pd
df = pd.DataFrame(results)
print(df)



# [optional] finish the wandb run, necessary in notebooks
wandb.finish()