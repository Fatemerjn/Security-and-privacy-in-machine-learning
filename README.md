# Security and Privacy in Machine Learning

A comprehensive implementation of security and privacy techniques in machine learning, including adversarial attacks, differential privacy, model extraction, data poisoning, and membership inference attacks.

## 🎯 Overview

This project explores various aspects of machine learning security and privacy through practical implementations and experiments. It covers both attack methods and defense mechanisms in the ML security domain.

## 📚 Project Structure

```
Security-and-privacy-in-machine-learning/
├── src/                           # Main source code
│   ├── adversarial/               # Adversarial attacks and defenses
│   ├── neural_networks/           # Neural network implementations from scratch
│   ├── differential_privacy/      # Differential privacy mechanisms
│   ├── model_extraction/          # Model extraction attacks
│   ├── poisoning/                 # Data poisoning attacks
│   └── membership_inference/      # Membership inference attacks
├── notebooks/                     # Original Jupyter notebooks
├── results/                       # Experimental results and outputs
│   ├── figures/                   # Generated plots and visualizations
│   └── models/                    # Trained models
├── data/                          # Datasets
├── tests/                         # Unit tests
├── docs/                          # Documentation
├── requirements.txt               # Python dependencies
├── setup.py                      # Package installation
└── README.md                     # Project documentation
```

## 🔬 Implemented Techniques

### 1. Adversarial Attacks
- **FGSM (Fast Gradient Sign Method)**: Basic adversarial example generation
- **PGD (Projected Gradient Descent)**: Iterative adversarial attacks
- **C&W Attack**: Confidence-based adversarial examples
- **Adversarial Training**: Defense mechanism through robust training

### 2. Neural Networks from Scratch
- **NumPy-based Neural Network**: Custom implementation without deep learning frameworks
- **Backpropagation**: Gradient computation and weight updates
- **Various Activation Functions**: ReLU, Sigmoid, Tanh implementations
- **Training Utilities**: Loss functions, optimizers, and evaluation metrics

### 3. Differential Privacy
- **Laplace Mechanism**: Adding calibrated noise for privacy
- **Gaussian Mechanism**: Alternative noise addition method
- **Privacy Budget Management**: ε-δ differential privacy guarantees
- **Private Training**: Differentially private machine learning
- **Composition Theorems**: Privacy budget tracking across queries

### 4. Model Extraction Attacks
- **Black-box Model Extraction**: Stealing model functionality through queries
- **Query Strategies**: Efficient querying for model replication
- **Substitute Model Training**: Creating surrogate models
- **Defense Mechanisms**: Query limiting and detection

### 5. Data Poisoning Attacks
- **Label Flipping**: Corrupting training labels
- **Backdoor Attacks**: Injecting hidden triggers
- **Clean-label Attacks**: Imperceptible poisoning
- **Targeted vs Indiscriminate**: Different attack objectives

### 6. Membership Inference Attacks
- **Shadow Model Training**: Creating proxy models for attack
- **Confidence-based Inference**: Using prediction confidence
- **Defense Strategies**: Regularization and privacy techniques
- **Evaluation Metrics**: Attack success measurement

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- NumPy
- Matplotlib
- scikit-learn
- TensorFlow/PyTorch (for comparison baselines)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Fatemerjn/Security-and-privacy-in-machine-learning.git
   cd Security-and-privacy-in-machine-learning
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install package in development mode:**
   ```bash
   pip install -e .
   ```

## 💻 Usage

### Running Individual Modules

#### Adversarial Attacks
```python
from src.adversarial import FGSM, PGD, CWAttack
from src.neural_networks import NeuralNetwork

# Load model and data
model = NeuralNetwork.load('results/models/mnist_model.pkl')
X_test, y_test = load_mnist_test()

# Generate adversarial examples
fgsm = FGSM(model, epsilon=0.1)
X_adv = fgsm.generate(X_test, y_test)
```

#### Differential Privacy
```python
from src.differential_privacy import LaplaceMechanism, PrivateTraining

# Apply differential privacy to training
private_trainer = PrivateTraining(epsilon=1.0, delta=1e-5)
private_model = private_trainer.fit(X_train, y_train)
```

#### Model Extraction
```python
from src.model_extraction import BlackBoxExtractor

# Extract model functionality
extractor = BlackBoxExtractor(target_model, query_budget=10000)
substitute_model = extractor.extract()
```

### Running Experiments

Each module includes example scripts demonstrating the techniques:

```bash
# Run adversarial attack experiments
python src/adversarial/experiments.py

# Run differential privacy experiments  
python src/differential_privacy/experiments.py

# Run model extraction experiments
python src/model_extraction/experiments.py
```

### Exporting Notebooks to Python Modules

The `tools/convert_notebooks.py` utility converts every notebook in the `notebooks/` directory into an executable Python module inside `src/`.

```bash
python tools/convert_notebooks.py
```

The generated files reproduce the original notebook code inside a `main()` function so you can run them with `python <module_path>.py`. The exporter strips notebook-only magics (e.g., `%matplotlib inline`) but otherwise preserves the original logic. Outputs are saved in the same locations the notebooks used (for example, model checkpoints next to the script).

## 📊 Results

Experimental results are automatically saved to the `results/` directory:
- **Figures**: Visualizations and plots in `results/figures/`
- **Models**: Trained models in `results/models/`
- **Logs**: Experimental logs and metrics

## 🔧 Configuration

Key parameters can be configured in each module:
- **Privacy Budget**: ε and δ values for differential privacy
- **Attack Strength**: ε for adversarial attacks, poison ratio for data poisoning
- **Model Architecture**: Hidden layers, activation functions, learning rates
- **Query Budgets**: Number of queries for extraction and inference attacks

