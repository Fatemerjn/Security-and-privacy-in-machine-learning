"""
Adversarial Attack Implementations

This module contains implementations of various adversarial attacks including:
- FGSM (Fast Gradient Sign Method)
- PGD (Projected Gradient Descent)  
- UAP (Universal Adversarial Perturbations)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
try:
    import torchattacks
except ImportError:
    print("Warning: torchattacks not available. Install with: pip install torchattacks")
    torchattacks = None


class FGSMAttack:
    """Fast Gradient Sign Method (FGSM) Attack Implementation"""
    
    def __init__(self, model, epsilon=8/255):
        """
        Initialize FGSM Attack
        
        Args:
            model: Target neural network model
            epsilon: Perturbation magnitude
        """
        self.model = model
        self.epsilon = epsilon
        
    def generate(self, images, labels, criterion=None):
        """
        Generate adversarial examples using FGSM
        
        Args:
            images: Input images tensor
            labels: True labels tensor  
            criterion: Loss function (default: CrossEntropyLoss)
            
        Returns:
            Adversarial examples tensor
        """
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
            
        images.requires_grad = True
        
        # Forward pass
        outputs = self.model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        self.model.zero_grad()
        loss.backward()
        
        # Generate adversarial examples
        data_grad = images.grad.data
        perturbed_images = images + self.epsilon * data_grad.sign()
        perturbed_images = torch.clamp(perturbed_images, 0, 1)
        
        return perturbed_images


class LinfPGDAttack:
    """Projected Gradient Descent (PGD) Attack Implementation"""
    
    def __init__(self, model, epsilon=8/255, k=4, alpha=2/255):
        """
        Initialize PGD Attack
        
        Args:
            model: Target neural network model
            epsilon: Maximum perturbation magnitude
            k: Number of iterations
            alpha: Step size per iteration
        """
        self.model = model
        self.epsilon = epsilon
        self.steps = k
        self.alpha = alpha

    def __call__(self, images, labels, criterion=None):
        """
        Generate adversarial examples using PGD
        
        Args:
            images: Input images tensor
            labels: True labels tensor
            criterion: Loss function (default: CrossEntropyLoss)
            
        Returns:
            Adversarial examples tensor
        """
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
            
        device = images.device
        perturbed_images = images.clone().detach().to(device)
        perturbed_images.requires_grad = True

        for _ in range(self.steps):
            outputs = self.model(perturbed_images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Update perturbed image with alpha step in gradient direction
            perturbed_images = perturbed_images + self.alpha * perturbed_images.grad.sign()
            perturbed_images = torch.clamp(perturbed_images, images - self.epsilon, images + self.epsilon)
            perturbed_images = torch.clamp(perturbed_images, 0, 1).detach_()
            perturbed_images.requires_grad = True

        return perturbed_images


class UAPAttack:
    """Universal Adversarial Perturbation (UAP) Attack Implementation"""
    
    def __init__(self, model, epsilon=8/255, delta=2/255, max_iters=10, data_loader=None):
        """
        Initialize UAP Attack
        
        Args:
            model: Target neural network model
            epsilon: Maximum perturbation magnitude
            delta: Step size for perturbation update
            max_iters: Number of iterations for UAP generation
            data_loader: DataLoader for training UAP
        """
        self.model = model
        self.epsilon = epsilon
        self.delta = delta
        self.max_iters = max_iters
        self.data_loader = data_loader
        self.uap = None

    def generate_uap(self, input_shape=(1, 3, 32, 32)):
        """
        Generate Universal Adversarial Perturbation
        
        Args:
            input_shape: Shape of input images (batch_size, channels, height, width)
            
        Returns:
            Generated UAP tensor
        """
        device = next(self.model.parameters()).device
        
        # Initialize universal perturbation
        self.uap = torch.zeros(input_shape).to(device)
        self.uap.requires_grad = True
        
        # Optimizer for UAP
        optimizer = optim.Adam([self.uap], lr=self.delta)
        criterion = nn.CrossEntropyLoss()
        
        self.model.eval()
        
        for itr in range(self.max_iters):
            print(f"UAP Generation Iteration {itr+1}/{self.max_iters}")
            for images, labels in tqdm(self.data_loader, desc=f"Iteration {itr+1}"):
                images, labels = images.to(device), labels.to(device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Apply current perturbation and clamp to [0,1]
                perturbed_images = torch.clamp(images + self.uap, 0, 1)
                
                # Forward pass
                outputs = self.model(perturbed_images)
                loss = criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                
                # Update perturbation
                optimizer.step()
                
                # Project to epsilon ball
                with torch.no_grad():
                    self.uap = torch.clamp(self.uap, -self.epsilon, self.epsilon)
                    self.uap.requires_grad = True
        
        return self.uap.detach()

    def __call__(self, images):
        """
        Apply UAP to input images
        
        Args:
            images: Input images tensor
            
        Returns:
            Perturbed images tensor
        """
        if self.uap is None:
            raise ValueError("UAP not generated. Call generate_uap() first.")
            
        return torch.clamp(images + self.uap, 0, 1)


def evaluate_attack(model, attack_fn, test_loader, device, attack_name="Attack"):
    """
    Evaluate attack success rate on test dataset
    
    Args:
        model: Target model
        attack_fn: Attack function that takes (images, labels) and returns adversarial images
        test_loader: Test data loader
        device: Device to run evaluation on
        attack_name: Name of attack for logging
        
    Returns:
        Attack success rate (accuracy on adversarial examples)
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc=f"Evaluating {attack_name}"):
            images, labels = images.to(device), labels.to(device)
            
            # Generate adversarial examples
            adv_images = attack_fn(images, labels)
            
            # Get predictions
            outputs = model(adv_images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy


def run_fgsm_evaluation(model, test_loader, device, epsilon_values=[1/255, 4/255, 8/255]):
    """
    Run FGSM attack evaluation for multiple epsilon values
    
    Args:
        model: Target model
        test_loader: Test data loader
        device: Device to run on
        epsilon_values: List of epsilon values to test
        
    Returns:
        Dictionary of epsilon -> accuracy mappings
    """
    results = {}
    
    for epsilon in epsilon_values:
        fgsm = FGSMAttack(model, epsilon=epsilon)
        accuracy = evaluate_attack(model, fgsm.generate, test_loader, device, 
                                 f"FGSM (eps={epsilon:.4f})")
        results[epsilon] = accuracy
        print(f"FGSM Attack with epsilon {epsilon:.4f}: Accuracy = {accuracy:.2f}%")
    
    return results


def run_pgd_evaluation(model, test_loader, device, k_values=[2, 4, 8], epsilon=8/255, alpha=2/255):
    """
    Run PGD attack evaluation for multiple k values
    
    Args:
        model: Target model
        test_loader: Test data loader
        device: Device to run on
        k_values: List of iteration counts to test
        epsilon: Perturbation magnitude
        alpha: Step size
        
    Returns:
        Dictionary of k -> accuracy mappings
    """
    results = {}
    
    for k in k_values:
        pgd = LinfPGDAttack(model, epsilon=epsilon, k=k, alpha=alpha)
        accuracy = evaluate_attack(model, pgd, test_loader, device, f"PGD (k={k})")
        results[k] = accuracy
        print(f"PGD Attack with k={k}: Accuracy = {accuracy:.2f}%")
    
    return results


# Wrapper for torchattacks FGSM if available
class TorchAttacksFGSM:
    """Wrapper for torchattacks FGSM implementation"""
    
    def __init__(self, model, epsilon=8/255):
        if torchattacks is None:
            raise ImportError("torchattacks not available. Install with: pip install torchattacks")
        
        self.attack = torchattacks.FGSM(model, eps=epsilon)
    
    def __call__(self, images, labels):
        return self.attack(images, labels)