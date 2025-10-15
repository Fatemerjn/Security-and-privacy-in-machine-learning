"""
Training utilities for adversarial learning

This module contains training functions for both standard and adversarial training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from .attacks import FGSMAttack, TorchAttacksFGSM


def train_standard(model, train_loader, test_loader=None, epochs=100, learning_rate=0.01, 
                  device="cpu", save_path=None):
    """
    Standard training procedure
    
    Args:
        model: Model to train
        train_loader: Training data loader
        test_loader: Test data loader (optional)
        epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device to train on
        save_path: Path to save model (optional)
        
    Returns:
        Trained model
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    
    model.train()
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # Calculate epoch metrics
        avg_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%")
        
        # Evaluate on test set periodically
        if test_loader is not None and (epoch + 1) % 10 == 0:
            test_acc = evaluate_model(model, test_loader, device)
            print(f"Test Accuracy after Epoch {epoch+1}: {test_acc:.2f}%")
    
    # Save model if path provided
    if save_path is not None:
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")
    
    return model


def train_adversarial(model, train_loader, test_loader=None, epochs=100, learning_rate=0.01,
                     epsilon=8/255, device="cpu", save_path=None, attack_type="fgsm"):
    """
    Adversarial training procedure
    
    Args:
        model: Model to train
        train_loader: Training data loader
        test_loader: Test data loader (optional)
        epochs: Number of training epochs
        learning_rate: Learning rate
        epsilon: Adversarial perturbation magnitude
        device: Device to train on
        save_path: Path to save model (optional)
        attack_type: Type of attack for adversarial training ("fgsm" or "torchattacks")
        
    Returns:
        Adversarially trained model
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    
    # Initialize attack for adversarial training
    if attack_type == "torchattacks":
        try:
            attack = TorchAttacksFGSM(model, epsilon=epsilon)
        except ImportError:
            print("torchattacks not available, using custom FGSM")
            attack = FGSMAttack(model, epsilon=epsilon)
    else:
        attack = FGSMAttack(model, epsilon=epsilon)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in tqdm(train_loader, desc=f"Adversarial Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            
            # Generate adversarial examples
            if hasattr(attack, 'generate'):
                adv_images = attack.generate(images, labels, criterion)
            else:
                adv_images = attack(images, labels)
            
            # Forward pass on adversarial images
            outputs = model(adv_images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # Calculate epoch metrics
        avg_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%")
        
        # Evaluate on clean test set periodically
        if test_loader is not None and (epoch + 1) % 10 == 0:
            test_acc = evaluate_model(model, test_loader, device)
            print(f"Clean Test Accuracy after Epoch {epoch+1}: {test_acc:.2f}%")
    
    # Save model if path provided
    if save_path is not None:
        torch.save(model.state_dict(), save_path)
        print(f"Adversarially trained model saved to {save_path}")
    
    return model


def evaluate_model(model, test_loader, device):
    """
    Evaluate model on clean test data
    
    Args:
        model: Model to evaluate
        test_loader: Test data loader
        device: Device to run evaluation on
        
    Returns:
        Test accuracy percentage
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy


def evaluate_robustness(model, test_loader, device, attacks=None):
    """
    Evaluate model robustness against multiple attacks
    
    Args:
        model: Model to evaluate
        test_loader: Test data loader
        device: Device to run evaluation on
        attacks: Dictionary of attack_name -> attack_function pairs
        
    Returns:
        Dictionary of attack results
    """
    if attacks is None:
        # Default attacks
        attacks = {
            "Clean": lambda x, y: x,  # No attack
            "FGSM_1": FGSMAttack(model, epsilon=1/255).generate,
            "FGSM_4": FGSMAttack(model, epsilon=4/255).generate,
            "FGSM_8": FGSMAttack(model, epsilon=8/255).generate,
        }
    
    results = {}
    
    for attack_name, attack_fn in attacks.items():
        correct = 0
        total = 0
        
        model.eval()
        for images, labels in tqdm(test_loader, desc=f"Evaluating {attack_name}"):
            images, labels = images.to(device), labels.to(device)
            
            # Apply attack (or no attack for clean)
            if attack_name == "Clean":
                test_images = images
            else:
                test_images = attack_fn(images, labels)
            
            with torch.no_grad():
                outputs = model(test_images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        results[attack_name] = accuracy
        print(f"{attack_name} Accuracy: {accuracy:.2f}%")
    
    return results


class EarlyStopping:
    """Early stopping utility for training"""
    
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_metric = None
        self.counter = 0
        self.best_weights = None
    
    def __call__(self, metric, model):
        """
        Check if training should stop
        
        Args:
            metric: Current metric value (higher is better)
            model: Current model
            
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_metric is None:
            self.best_metric = metric
            self.best_weights = model.state_dict().copy() if self.restore_best_weights else None
        elif metric < self.best_metric + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights and self.best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_metric = metric
            self.counter = 0
            self.best_weights = model.state_dict().copy() if self.restore_best_weights else None
        
        return False