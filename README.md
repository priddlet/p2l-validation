# Pick-To-Learn, in Python / Jax

A JAX implementation of the Pick-to-Learn (P2L) algorithm from the paper:
["The Pick-to-Learn Algorithm: Empowering Compression for Tight Generalization Bounds and Improved Post-training Performance"](https://proceedings.neurips.cc/paper_files/paper/2023/file/3a4f287883609241031e6818bd01133e-Paper-Conference.pdf) -- Paccagnan, Campi, Garatti (2023)

## Installation

A virtual environment is optional, but highly recommended.

```bash
git clone https://github.com/danielpmorton/picktolearn
cd picktolearn
pip install -e .
```

For JAX GPU support, you'll need to run:
```bash
pip install -U "jax[cuda12]"
```

To run the examples and evaluation framework, use:
```bash
pip install -e ".[examples]"
```

## Quick Start

### Basic P2L Usage

```python
from picktolearn.mnist_example import BinaryMNISTP2LConfig
from picktolearn.p2l import pick_to_learn
import jax

# Create configuration
config = BinaryMNISTP2LConfig(
    n_datapoints=1000,
    dataset_slice_index=0,
    dropout_prob=0.2,
    learning_rate=0.01,
    momentum=0.95,
    convergence_param=0.69314718,  # -ln(0.5)
    pretrain_fraction=0.5,
    max_iterations=1000,
    train_epochs=200,
    batch_size=1000,
    confidence_param=0.035,
)

# Run P2L
key = jax.random.key(42)
result = pick_to_learn(config, key)

print(f"Converged: {result['converged']}")
print(f"Iterations: {result['num_iterations']}")
print(f"Support set size: {len(result['support_indices'])}")
print(f"Generalization bound: {result['generalization_bound']:.5f}")
```

### Running the Evaluation Framework

The evaluation framework provides comprehensive validation and analysis of the P2L implementation:

```bash
# Run all evaluation modes (validation, replication, analysis)
python picktolearn/run_evaluation.py --mode all

# Run only validation tests
python picktolearn/run_evaluation.py --mode validation

# Run paper replication experiments
python picktolearn/run_evaluation.py --mode replication

# Run comprehensive analysis
python picktolearn/run_evaluation.py --mode analysis
```

## Evaluation Framework

The evaluation framework consists of three main components:

### 1. Validation (`validation.py`)

Validates the P2L implementation by:
- Testing basic functionality
- Checking generalization bound calculations
- Analyzing convergence properties
- Comparing with expected theoretical properties

### 2. Replication (`evaluation.py`)

Replicates the original paper experiments:
- Binary MNIST classification
- Synthetic regression problem
- Comparison with baseline methods
- Generation of paper-style plots

### 3. Analysis (`analysis.py`)

Provides detailed analysis of P2L behavior:
- Convergence analysis and visualization
- Parameter sensitivity studies
- Support set analysis
- Generalization bound analysis
- Comparison with theoretical properties

## Understanding P2L Results

### Key Metrics

1. **Convergence**: Whether P2L converged within the maximum iterations
2. **Support Set Size**: Number of examples in the final support set
3. **Generalization Bound**: Theoretical bound on generalization error
4. **Iterations**: Number of P2L iterations performed

### Interpreting Results

- **Good convergence**: P2L should converge in most cases (>50% of runs)
- **Reasonable support size**: Should be smaller than the full dataset but not too small
- **Valid generalization bound**: Should be between 0.1 and 0.9 for typical problems
- **Stable results**: Multiple runs should produce similar results

### Expected Behavior

Based on the original paper:
- P2L should converge for most reasonable problems
- Support sets should be informative but smaller than full datasets
- Generalization bounds should be tighter than test-set bounds
- Results should be consistent across multiple runs

## Troubleshooting

### Common Issues

1. **P2L doesn't converge**: Try adjusting `convergence_param` or `max_iterations`
2. **Poor generalization bounds**: Check that `confidence_param` is reasonable (0.01-0.1)
3. **Inconsistent results**: Ensure proper random seed setting
4. **Memory issues**: Reduce `batch_size` or `n_datapoints`

### Performance Tips

- Use GPU acceleration for faster training
- Reduce `train_epochs` for faster experimentation
- Use smaller datasets for quick testing
- Adjust `max_iterations` based on problem complexity

## File Structure

```
picktolearn/
├── p2l.py              # Core P2L implementation
├── mnist_example.py    # Binary MNIST example
├── evaluation.py       # Paper replication framework
├── validation.py       # Implementation validation
├── analysis.py         # Comprehensive analysis tools
├── run_evaluation.py   # Main evaluation script
└── results/            # Generated plots and results
```

## Contributing

When contributing to the P2L implementation:

1. Run the validation suite: `python picktolearn/run_evaluation.py --mode validation`
2. Ensure paper replication works: `python picktolearn/run_evaluation.py --mode replication`
3. Check that analysis results are reasonable: `python picktolearn/run_evaluation.py --mode analysis`
4. Add tests for new functionality

## Citation

If you use this implementation in your research, please cite the original paper:

```bibtex
@inproceedings{paccagnan2023pick,
  title={The Pick-to-Learn Algorithm: Empowering Compression for Tight Generalization Bounds and Improved Post-training Performance},
  author={Paccagnan, Dario and Campi, Marco C and Garatti, Simone},
  booktitle={Advances in Neural Information Processing Systems},
  year={2023}
}
```
