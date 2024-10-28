from torch.utils.data import DataLoader, random_split
from torchvision.transforms import v2
from TemporalImageDataset import TemporalImageDataset

transform = v2.Compose([
    v2.RandomRotation(degrees=30),
    v2.ToImage()
])

dataset = TemporalImageDataset(root_dir='Dataset\L15-1669E-1153N_6678_3579_13\images', transform=transform)

train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
val_loader = DataLoader(val_set, batch_size=4, shuffle=False)
test_loader = DataLoader(test_set, batch_size=4, shuffle=False)

def test_dataloader(dataloader, name):
    print(f"Testing {name}:")
    for i, (start_images, end_images, time_skips) in enumerate(dataloader):
        print(f"Batch {i + 1}:")
        print(f"  Start Images Shape: {start_images.shape}") 
        print(f"  End Images Shape: {end_images.shape}")      
        print(f"  Time Skips: {time_skips}")              
        
    print("")

test_dataloader(train_loader, "Train DataLoader")
test_dataloader(val_loader, "Validation DataLoader")
test_dataloader(test_loader, "Test DataLoader")