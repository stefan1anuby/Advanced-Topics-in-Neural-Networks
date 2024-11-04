import torch.optim as optim

def get_optimizer(config, model):
    if config['optimizer'] == "SGD":
        return optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=config.get('momentum', 0.9), weight_decay=config.get('weight_decay', 0.0))
    elif config['optimizer'] == "Adam":
        return optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config.get('weight_decay', 0.0))
    elif config['optimizer'] == "AdamW":
        return optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config.get('weight_decay', 0.0))
    elif config['optimizer'] == "RmsProp":
        return optim.RMSprop(model.parameters(), lr=config['learning_rate'], weight_decay=config.get('weight_decay', 0.0))
    else:
        raise ValueError("Unsupported optimizer specified!")
