# Results Directory

This directory contains all experimental results, trained models, and generated outputs from the machine learning security and privacy experiments.

## Directory Structure

### `figures/`
Contains all generated plots, visualizations, and figures:
- Attack success rate comparisons
- Model performance metrics
- Privacy-utility trade-off plots
- Adversarial example visualizations
- Training loss and accuracy curves

### `models/`
Stores trained model checkpoints and weights:
- Standard trained models
- Adversarially trained models
- Shadow models for membership inference
- Substitute models from extraction attacks
- Private models with differential privacy

### `logs/`
Experiment logs and detailed results:
- Training logs with loss/accuracy over time
- Attack evaluation results
- Privacy analysis reports
- Hyperparameter search results
- Benchmark comparisons

### `experiments/`
Organized experimental results by technique:
- `adversarial/` - FGSM, PGD, UAP attack results
- `differential_privacy/` - Privacy mechanism evaluations
- `membership_inference/` - Attack and defense results
- `model_extraction/` - Extraction attack outcomes
- `poisoning/` - Data poisoning experiment results

## File Naming Conventions

### Models
- `{architecture}_{dataset}_{method}_{timestamp}.pth`
- Example: `resnet18_cifar10_adversarial_20231015.pth`

### Figures
- `{experiment}_{metric}_{parameters}_{timestamp}.png`
- Example: `fgsm_accuracy_eps0.1_20231015.png`

### Logs
- `{experiment}_{timestamp}.log`
- Example: `adversarial_training_20231015.log`

## Usage Notes

- Results are automatically saved by experiment scripts
- Timestamps ensure unique filenames and version tracking
- Large model files (>100MB) should be compressed or stored separately
- Figures are saved in high resolution (300 DPI) for publications
- JSON files contain structured experiment metadata and results

## Cleanup

Old results can be cleaned up periodically:
```bash
# Remove results older than 30 days
find results/ -type f -mtime +30 -delete
```

## Backup

Important results should be backed up regularly:
- Use version control for small result files
- Archive large model files to external storage
- Document significant experimental findings