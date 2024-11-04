import torch.optim as optim

def get_scheduler(config, optimizer):
    if config['scheduler'] == "StepLR":
        return optim.lr_scheduler.StepLR(optimizer, step_size=config.get('step_size', 30), gamma=config.get('gamma', 0.1))
    elif config['scheduler'] == "ReduceLROnPlateau":
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=config.get('factor', 0.1), patience=config.get('patience', 10))
    elif config['scheduler'] is None:
        return None
    else:
        raise ValueError("Unsupported scheduler specified!")
