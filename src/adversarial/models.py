"""
Neural Network Models for Adversarial Learning

This module contains model architectures used for adversarial experiments.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18


class ResNet(nn.Module):
    """Custom ResNet18 model for CIFAR-10"""
    
    def __init__(self, num_classes=10):
        """
        Initialize ResNet model
        
        Args:
            num_classes: Number of output classes
        """
        super(ResNet, self).__init__()
        # Use ResNet18 backbone without final layers
        self.conv = nn.Sequential(*list(resnet18(weights=None).children())[:-2])
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor
            
        Returns:
            Logits tensor
        """
        x = self.conv(x)
        x = torch.flatten(x, start_dim=1)
        logits = self.fc(x)
        return logits


class ResNetAdversarial(nn.Module):
    """ResNet model specifically for adversarial training"""
    
    def __init__(self, num_classes=10):
        """
        Initialize adversarial ResNet model
        
        Args:
            num_classes: Number of output classes
        """
        super(ResNetAdversarial, self).__init__()
        self.conv = nn.Sequential(*list(resnet18(weights=None).children())[:-2])
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor
            
        Returns:
            Logits tensor
        """
        x = self.conv(x)
        x = torch.flatten(x, start_dim=1)
        logits = self.fc(x)
        return logits


def create_model(model_type="resnet", num_classes=10, device="cpu"):
    """
    Factory function to create models
    
    Args:
        model_type: Type of model ("resnet" or "resnet_adv")
        num_classes: Number of output classes
        device: Device to place model on
        
    Returns:
        Model instance
    """
    if model_type == "resnet":
        model = ResNet(num_classes=num_classes)
    elif model_type == "resnet_adv":
        model = ResNetAdversarial(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model.to(device)


def load_model(model_path, model_type="resnet", num_classes=10, device="cpu"):
    """
    Load a saved model
    
    Args:
        model_path: Path to saved model weights
        model_type: Type of model architecture
        num_classes: Number of output classes
        device: Device to place model on
        
    Returns:
        Loaded model instance
    """
    model = create_model(model_type, num_classes, device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model


def save_model(model, model_path):
    """
    Save model weights
    
    Args:
        model: Model instance to save
        model_path: Path where to save the model
    """
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")