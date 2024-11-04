import torch
from utils import load_config, get_device
from datasets import get_dataset
from models import load_model

from optimizers import get_optimizer
from schedulers import get_scheduler
from early_stopping import EarlyStopping

def train_one_epoch(model, train_loader, optimizer, device, criterion):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    return running_loss / len(train_loader)

def validate(model, val_loader, device, criterion):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
    return running_loss / len(val_loader)

def main():
    # load config
    config = load_config()
    
    # set device
    device = get_device(config['device'])
    
    # load data
    train_loader, test_loader = get_dataset(config['dataset'], config['batch_size'], use_cache=False) #set use_cache to True for using the cache but I/O operations are prettly slow appareantly
    
    # load model
    model = load_model(config['model'], num_classes=10)
    model = model.to(device)
    
    # initialize optimizer and scheduler
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)
    
    # initialize early stopping
    early_stopping = EarlyStopping(patience=config.get('early_stopping_patience', 5))
    
    # define loss function
    criterion = torch.nn.CrossEntropyLoss()
    
    print("Starting training...")
    for epoch in range(config['epochs']):
        # training phase
        train_loss = train_one_epoch(model, train_loader, optimizer, device, criterion)
        
        # validation phase
        val_loss = validate(model, test_loader, device, criterion)
        
        print(f"Epoch {epoch+1}/{config['epochs']}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # check early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
        
        # step the scheduler if applicable
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
    
if __name__ == "__main__":
    main()
