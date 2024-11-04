# CIFAR-100 Parameter Sweep Experiment

This project explores different configurations of model architectures, data augmentation techniques and optimizers. The experiment was conducted using wandb for tracking and logging, with each configuration logged as a separate run.

## Parameters

The following parameters were varied during the sweep to observe their impact on model performance:

### 1. Model Architecture
   - **ResNet18**: 
   - **VGG16**:

### 2. Data Augmentation Technique
   - **Basic**:
     - **RandomHorizontalFlip**: Flips the image horizontally with a 50% probability.
     - **RandomCrop(32, padding=4)**: Adds a 4-pixel padding around the image and crops a 32x32 random section.
   
   - **Advanced**:
     - **RandomHorizontalFlip**
     - **RandomCrop(32, padding=4)**
     - **ColorJitter**: Randomly adjusts brightness, contrast, saturation, and hue.
     - **RandomRotation(15)**: Rotates the image randomly by up to ±15 degrees.

### 3. Optimizer
   - **SGD**
   - **Adam**

---

## Results

Each configuration was run with a unique combination of these parameters, and metrics such as training accuracy, test accuracy, and loss were logged separately for each run. Results are recorded and available on the wandb dashboard.

### Summary of Results

| Model     | Augmentation | Optimizer | Test Accuracy | Link |
|-----------|--------------|-----------|---------------|-------------|
| ResNet18  | Basic        | SGD       | 0.27         | [Link](https://wandb.ai/stefaneduard2002-universitatea-alexandru-ioan-cuza-din-ia-i/ResNet18_basic_SGD_2024-11-03_21-05-01?nw=nwuserstefaneduard2002) |
| ResNet18  | Basic        | Adam       | 0.28         | [Link](https://wandb.ai/stefaneduard2002-universitatea-alexandru-ioan-cuza-din-ia-i/ResNet18_basic_Adam_2024-11-03_21-10-01?nw=nwuserstefaneduard2002) |
| ResNet18  | Advanced     | SGD      | 0.25         | [Link](https://wandb.ai/stefaneduard2002-universitatea-alexandru-ioan-cuza-din-ia-i/ResNet18_advanced_SGD_2024-11-03_21-15-28?nw=nwuserstefaneduard2002) |
| ResNet18     | Advanced     | Adam      | 0.26         | [Link](https://wandb.ai/stefaneduard2002-universitatea-alexandru-ioan-cuza-din-ia-i/ResNet18_advanced_Adam_2024-11-04_09-38-22?nw=nwuserstefaneduard2002) |
| VGG16     | Basic     | SGD      | 0.12         | [Link](https://wandb.ai/stefaneduard2002-universitatea-alexandru-ioan-cuza-din-ia-i/VGG16_basic_SGD_2024-11-04_08-18-48?nw=nwuserstefaneduard2002) |
| VGG16     | Basic     | Adam      | 0.01         | [Link](https://wandb.ai/stefaneduard2002-universitatea-alexandru-ioan-cuza-din-ia-i/VGG16_basic_Adam_2024-11-04_08-30-28?nw=nwuserstefaneduard2002) |
| VGG16     | Advanced     | SGD      | 0.01         | [Link](https://wandb.ai/stefaneduard2002-universitatea-alexandru-ioan-cuza-din-ia-i/VGG16_advanced_SGD_2024-11-04_08-47-56?nw=nwuserstefaneduard2002) |
| VGG16     | Advanced     | Adam      | 0.01         | [Link](https://wandb.ai/stefaneduard2002-universitatea-alexandru-ioan-cuza-din-ia-i/VGG16_advanced_Adam_2024-11-04_09-05-47?nw=nwuserstefaneduard2002) |


> Note: For detailed graphs and metrics for each configuration, please check each configuration link.
