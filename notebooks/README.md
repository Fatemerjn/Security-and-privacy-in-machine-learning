# Jupyter Notebooks

This directory contains interactive Jupyter notebooks demonstrating various security and privacy techniques in machine learning.

## Notebooks Overview

### 1. `adversarial_attacks_and_defenses.ipynb`
**Topics:** FGSM, PGD, UAP attacks and adversarial training
- Implementation of Fast Gradient Sign Method (FGSM)
- Projected Gradient Descent (PGD) attacks from scratch
- Universal Adversarial Perturbations (UAP)
- Adversarial training for robust models
- Visualization of attack effects
- Comparison between standard and adversarially trained models

### 2. `neural_networks_from_scratch.ipynb` 
**Topics:** NumPy-based neural network implementation
- Custom neural network implementation without deep learning frameworks
- Backpropagation algorithm from scratch
- Various activation functions (ReLU, Sigmoid, Tanh)
- Training and evaluation utilities
- Gradient computation and weight updates

### 3. `differential_privacy_mechanisms.ipynb`
**Topics:** Privacy-preserving machine learning
- Laplace and Gaussian noise mechanisms
- ε-δ differential privacy guarantees
- Private training algorithms
- Privacy budget management and composition
- Trade-offs between privacy and utility

### 4. `membership_inference_attacks.ipynb`
**Topics:** Privacy attacks on ML models
- Shadow model training for membership inference
- Confidence-based attack strategies
- Defense mechanisms against inference attacks
- Evaluation metrics for privacy breaches
- Impact of model overfitting on privacy

### 5. `model_extraction_attacks.ipynb`
**Topics:** Stealing model functionality
- Black-box model extraction techniques
- Query strategies for efficient extraction
- Substitute model training
- Defense mechanisms (query limiting, detection)
- Evaluation of extraction success

### 6. `data_poisoning_attacks.ipynb`
**Topics:** Training data manipulation attacks
- Label flipping attacks
- Backdoor injection techniques
- Clean-label poisoning methods
- Targeted vs. indiscriminate attacks
- Detection and mitigation strategies

## Running the Notebooks

1. **Install dependencies:**
   ```bash
   pip install -r ../requirements.txt
   ```

2. **Start Jupyter:**
   ```bash
   jupyter lab
   # or
   jupyter notebook
   ```

3. **Navigate to this directory** and open any notebook

## Data Requirements

- Most notebooks use standard datasets (CIFAR-10, MNIST, etc.)
- Data will be automatically downloaded when needed
- Custom datasets should be placed in the `../data/` directory

## Results

- Generated plots and figures are saved to `../results/figures/`
- Trained models are saved to `../results/models/`
- Experiment logs and metrics are saved to `../results/logs/`

## Notes

- Each notebook is self-contained with necessary imports
- Code from notebooks has been extracted to `../src/` modules for reusability
- Run notebooks in order for better understanding of concepts
- Some experiments may take significant time to complete