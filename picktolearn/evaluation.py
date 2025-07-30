"""Evaluation framework for P2L to replicate original paper results

This module provides tools to:
1. Run P2L on classification and regression tasks
2. Compare with test-set and PAC-Bayes bounds
3. Generate plots similar to the original paper
4. Validate P2L implementation correctness

Based on the paper: "The Pick-to-Learn Algorithm: Empowering Compression for Tight 
Generalization Bounds and Improved Post-training Performance" -- Paccagnan, Campi, Garatti (2023)
"""

import math
import numpy as np
import jax
import jax.numpy as jnp
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from abc import ABC, abstractmethod

from p2l import P2LConfig, pick_to_learn
from mnist_example import BinaryMNISTP2LConfig
import optax
from flax import nnx


@dataclass
class EvaluationResult:
    """Container for evaluation results"""
    method: str
    generalization_bound: float
    test_accuracy: float
    support_set_size: int
    training_time: float
    metadata: Dict[str, Any]


class BaselineMethod(ABC):
    """Abstract base class for baseline methods"""
    
    @abstractmethod
    def run(self, data: jax.Array, targets: jax.Array, key: jax.Array) -> EvaluationResult:
        """Run the baseline method"""
        pass


class TestSetBaseline(BaselineMethod):
    """Test-set based generalization bound"""
    
    def __init__(self, test_fraction: float = 0.2):
        self.test_fraction = test_fraction
    
    def run(self, data: jax.Array, targets: jax.Array, key: jax.Array) -> EvaluationResult:
        # Split data into train/test
        n_total = data.shape[0]
        n_test = int(n_total * self.test_fraction)
        perm = jax.random.permutation(key, n_total)
        
        test_indices = perm[:n_test]
        train_indices = perm[n_test:]
        
        # Use train data for P2L
        train_data = data[train_indices]
        train_targets = targets[train_indices]
        test_data = data[test_indices]
        test_targets = targets[test_indices]
        
        # Run P2L on training data
        # Note: This is a simplified version - in practice you'd need to adapt the config
        # to work with the reduced dataset size
        raise NotImplementedError("Test-set baseline needs proper implementation")


class PACBayesBaseline(BaselineMethod):
    """PAC-Bayes based generalization bound"""
    
    def __init__(self, prior_std: float = 0.02, dropout_prob: float = 0.1):
        self.prior_std = prior_std
        self.dropout_prob = dropout_prob
    
    def run(self, data: jax.Array, targets: jax.Array, key: jax.Array) -> EvaluationResult:
        # PAC-Bayes implementation would go here
        # This is a placeholder - full implementation would be quite complex
        raise NotImplementedError("PAC-Bayes baseline needs proper implementation")


class SyntheticRegressionConfig(P2LConfig):
    """P2L Configuration for synthetic regression problem from the original paper"""
    
    def __init__(
        self,
        n_datapoints: int,
        noise_std: float,
        learning_rate: float,
        momentum: float,
        convergence_param: float,
        pretrain_fraction: float,
        max_iterations: int,
        train_epochs: int,
        batch_size: int,
        confidence_param: float,
    ):
        self.n_datapoints = n_datapoints
        self.noise_std = noise_std
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.convergence_param = convergence_param
        
        super().__init__(
            pretrain_fraction,
            max_iterations,
            train_epochs,
            batch_size,
            confidence_param,
        )
    
    def init_model(self, key):
        return SyntheticRegressionModel(key=key)
    
    def init_optimizer(self):
        return optax.sgd(self.learning_rate, self.momentum)
    
    def init_data(self, key):
        # Generate synthetic regression data as described in the paper
        # y = sin(2πx) + ε, where ε ~ N(0, noise_std^2)
        x_key, y_key = jax.random.split(key)
        
        x = jax.random.uniform(x_key, (self.n_datapoints, 1), minval=0, maxval=1)
        true_y = jnp.sin(2 * jnp.pi * x)
        noise = jax.random.normal(y_key, (self.n_datapoints, 1)) * self.noise_std
        y = true_y + noise
        
        return x, y.squeeze()
    
    def loss_function(self, model_output, target):
        return jnp.mean((model_output - target) ** 2)
    
    def accuracy(self, model_output, target):
        # For regression, use R² score as "accuracy"
        ss_res = jnp.sum((target - model_output) ** 2)
        ss_tot = jnp.sum((target - jnp.mean(target)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        return r2
    
    def eval_p2l_convergence(self, model_output, target):
        # For regression, check if max prediction error is below threshold
        errors = jnp.abs(model_output - target)
        max_error_index = jnp.argmax(errors)
        converged = errors[max_error_index] <= self.convergence_param
        return max_error_index, converged


class SyntheticRegressionModel(nnx.Module):
    """Simple neural network for synthetic regression"""
    
    def __init__(self, key: jax.Array):
        param_keys = jax.random.split(key, 3)
        self.l1 = nnx.Linear(1, 50, rngs=nnx.Rngs(params=param_keys[0]))
        self.l2 = nnx.Linear(50, 50, rngs=nnx.Rngs(params=param_keys[1]))
        self.l3 = nnx.Linear(50, 1, rngs=nnx.Rngs(params=param_keys[2]))
    
    def __call__(self, x, deterministic: bool, key: jax.Array):
        x = self.l1(x)
        x = jax.nn.relu(x)
        x = self.l2(x)
        x = jax.nn.relu(x)
        x = self.l3(x)
        return x.squeeze()


class P2LEvaluator:
    """Main evaluator class for P2L experiments"""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.key = jax.random.key(seed)
    
    def run_classification_experiment(
        self, 
        config: P2LConfig,
        n_runs: int = 5
    ) -> List[Dict[str, Any]]:
        """Run classification experiment multiple times"""
        results = []
        
        for run in range(n_runs):
            print(f"\n=== Classification Run {run + 1}/{n_runs} ===")
            key, self.key = jax.random.split(self.key)
            result = pick_to_learn(config, key)
            results.append(result)
        
        return results
    
    def run_regression_experiment(
        self,
        config: P2LConfig,
        n_runs: int = 5
    ) -> List[Dict[str, Any]]:
        """Run regression experiment multiple times"""
        results = []
        
        for run in range(n_runs):
            print(f"\n=== Regression Run {run + 1}/{n_runs} ===")
            key, self.key = jax.random.split(self.key)
            result = pick_to_learn(config, key)
            results.append(result)
        
        return results
    
    def compare_with_baselines(
        self,
        p2l_results: List[Dict[str, Any]],
        baseline_methods: List[BaselineMethod],
        data: jax.Array,
        targets: jax.Array
    ) -> Dict[str, List[EvaluationResult]]:
        """Compare P2L results with baseline methods"""
        comparison_results = {
            'p2l': [],
            'baselines': {}
        }
        
        # Convert P2L results to EvaluationResult format
        for result in p2l_results:
            eval_result = EvaluationResult(
                method='P2L',
                generalization_bound=result['generalization_bound'],
                test_accuracy=result.get('test_accuracy', 0.0),  # Would need to be computed
                support_set_size=len(result['support_indices']),
                training_time=0.0,  # Would need to be tracked
                metadata=result
            )
            comparison_results['p2l'].append(eval_result)
        
        # Run baseline methods
        for baseline in baseline_methods:
            baseline_name = baseline.__class__.__name__
            comparison_results['baselines'][baseline_name] = []
            
            for run in range(len(p2l_results)):
                key, self.key = jax.random.split(self.key)
                try:
                    baseline_result = baseline.run(data, targets, key)
                    comparison_results['baselines'][baseline_name].append(baseline_result)
                except NotImplementedError:
                    print(f"Warning: {baseline_name} not implemented yet")
        
        return comparison_results
    
    def plot_results(
        self,
        results: List[Dict[str, Any]],
        task_type: str = "classification",
        save_path: Optional[str] = None
    ):
        """Generate plots similar to the original paper"""
        
        # Extract metrics
        bounds = [r['generalization_bound'] for r in results]
        support_sizes = [len(r['support_indices']) for r in results]
        iterations = [r['num_iterations'] for r in results]
        converged = [r['converged'] for r in results]
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Generalization bounds distribution
        axes[0, 0].hist(bounds, bins=10, alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel('Generalization Bound')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of Generalization Bounds')
        axes[0, 0].axvline(np.mean(bounds), color='red', linestyle='--', 
                           label=f'Mean: {np.mean(bounds):.4f}')
        axes[0, 0].legend()
        
        # Plot 2: Support set size vs iterations
        axes[0, 1].scatter(iterations, support_sizes, alpha=0.7)
        axes[0, 1].set_xlabel('P2L Iterations')
        axes[0, 1].set_ylabel('Support Set Size')
        axes[0, 1].set_title('Convergence Behavior')
        
        # Plot 3: Loss progression (if available)
        if 'losses' in results[0]:
            # Plot loss progression for first few runs
            for i, result in enumerate(results[:3]):
                losses = result['losses']
                axes[1, 0].plot(losses, label=f'Run {i+1}', alpha=0.7)
            axes[1, 0].set_xlabel('P2L Iteration')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].set_title('Loss Progression')
            axes[1, 0].legend()
        
        # Plot 4: Convergence statistics
        converged_count = sum(converged)
        total_count = len(converged)
        axes[1, 1].pie([converged_count, total_count - converged_count], 
                       labels=['Converged', 'Not Converged'],
                       autopct='%1.1f%%')
        axes[1, 1].set_title('Convergence Statistics')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_paper_style_plots(
        self,
        classification_results: List[Dict[str, Any]],
        regression_results: List[Dict[str, Any]],
        save_dir: Optional[str] = None
    ):
        """Generate plots in the style of the original paper"""
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # Figure 1: Comparison of generalization bounds
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Classification bounds
        class_bounds = [r['generalization_bound'] for r in classification_results]
        ax1.hist(class_bounds, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Generalization Bound')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Binary MNIST Classification\nGeneralization Bounds Distribution')
        ax1.axvline(np.mean(class_bounds), color='red', linestyle='--', 
                    label=f'Mean: {np.mean(class_bounds):.4f}')
        ax1.legend()
        
        # Regression bounds
        reg_bounds = [r['generalization_bound'] for r in regression_results]
        ax2.hist(reg_bounds, bins=15, alpha=0.7, color='lightcoral', edgecolor='black')
        ax2.set_xlabel('Generalization Bound')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Synthetic Regression\nGeneralization Bounds Distribution')
        ax2.axvline(np.mean(reg_bounds), color='red', linestyle='--', 
                    label=f'Mean: {np.mean(reg_bounds):.4f}')
        ax2.legend()
        
        plt.tight_layout()
        
        if save_dir:
            plt.savefig(f"{save_dir}/generalization_bounds_comparison.png", 
                       dpi=300, bbox_inches='tight')
        
        plt.show()
        
        # Figure 2: Support set size progression
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Classification support sizes
        class_sizes = [len(r['support_indices']) for r in classification_results]
        ax1.hist(class_sizes, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Final Support Set Size')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Binary MNIST Classification\nSupport Set Size Distribution')
        ax1.axvline(np.mean(class_sizes), color='red', linestyle='--', 
                    label=f'Mean: {np.mean(class_sizes):.1f}')
        ax1.legend()
        
        # Regression support sizes
        reg_sizes = [len(r['support_indices']) for r in regression_results]
        ax2.hist(reg_sizes, bins=15, alpha=0.7, color='lightcoral', edgecolor='black')
        ax2.set_xlabel('Final Support Set Size')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Synthetic Regression\nSupport Set Size Distribution')
        ax2.axvline(np.mean(reg_sizes), color='red', linestyle='--', 
                    label=f'Mean: {np.mean(reg_sizes):.1f}')
        ax2.legend()
        
        plt.tight_layout()
        
        if save_dir:
            plt.savefig(f"{save_dir}/support_set_sizes.png", dpi=300, bbox_inches='tight')
        
        plt.show()


def create_paper_experiments():
    """Create experiment configurations matching the original paper"""
    
    # Classification experiment (Binary MNIST)
    classification_config = BinaryMNISTP2LConfig(
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
    
    # Regression experiment (Synthetic)
    regression_config = SyntheticRegressionConfig(
        n_datapoints=200,
        noise_std=0.1,
        learning_rate=0.01,
        momentum=0.9,
        convergence_param=0.1,
        pretrain_fraction=0.5,
        max_iterations=1000,
        train_epochs=100,
        batch_size=200,
        confidence_param=0.035,
    )
    
    return classification_config, regression_config


def run_paper_replication():
    """Run the main paper replication experiment"""
    
    print("=== P2L Paper Replication Experiment ===")
    
    # Create evaluator
    evaluator = P2LEvaluator(seed=42)
    
    # Create experiment configurations
    class_config, reg_config = create_paper_experiments()
    
    # Run classification experiments
    print("\nRunning classification experiments...")
    classification_results = evaluator.run_classification_experiment(
        class_config, n_runs=5
    )
    
    # Run regression experiments
    print("\nRunning regression experiments...")
    regression_results = evaluator.run_regression_experiment(
        reg_config, n_runs=5
    )
    
    # Generate plots
    print("\nGenerating plots...")
    evaluator.generate_paper_style_plots(
        classification_results, 
        regression_results,
        save_dir="./results"
    )
    
    # Print summary statistics
    print("\n=== SUMMARY STATISTICS ===")
    
    class_bounds = [r['generalization_bound'] for r in classification_results]
    reg_bounds = [r['generalization_bound'] for r in regression_results]
    
    print(f"Classification - Mean bound: {np.mean(class_bounds):.4f} ± {np.std(class_bounds):.4f}")
    print(f"Regression - Mean bound: {np.mean(reg_bounds):.4f} ± {np.std(reg_bounds):.4f}")
    
    class_sizes = [len(r['support_indices']) for r in classification_results]
    reg_sizes = [len(r['support_indices']) for r in regression_results]
    
    print(f"Classification - Mean support size: {np.mean(class_sizes):.1f} ± {np.std(class_sizes):.1f}")
    print(f"Regression - Mean support size: {np.mean(reg_sizes):.1f} ± {np.std(reg_sizes):.1f}")
    
    return classification_results, regression_results


if __name__ == "__main__":
    run_paper_replication() 