"""Validation script for P2L implementation

This script provides comprehensive validation of the P2L implementation by:
1. Testing basic functionality
2. Comparing with expected behaviors from the original paper
3. Validating convergence properties
4. Checking generalization bound calculations
"""

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from typing import Dict, List, Any
import time

from p2l import P2LConfig, pick_to_learn, generalization_bound
from mnist_example import BinaryMNISTP2LConfig
from evaluation import SyntheticRegressionConfig, P2LEvaluator
from flax import nnx


class SimpleTestModel(nnx.Module):
    """Simple model for validation testing"""
    
    def __init__(self, key: jax.Array):
        self.linear = nnx.Linear(2, 2, rngs=nnx.Rngs(params=key))
    
    def __call__(self, x, deterministic: bool, key: jax.Array):
        return self.linear(x)


class P2LValidator:
    """Validator for P2L implementation"""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.key = jax.random.key(seed)
    
    def test_basic_functionality(self) -> Dict[str, Any]:
        """Test basic P2L functionality"""
        print("=== Testing Basic P2L Functionality ===")
        
        # Create a simple synthetic dataset for testing
        key, self.key = jax.random.split(self.key)
        
        # Simple 2D classification problem
        n_samples = 100
        x = jax.random.normal(key, (n_samples, 2))
        y = (x[:, 0] + x[:, 1] > 0).astype(jnp.int32)
        
        # Create a simple config for testing
        class SimpleTestConfig(P2LConfig):
            def __init__(self):
                super().__init__(
                    pretrain_fraction=0.3,
                    max_iterations=50,
                    train_epochs=10,
                    batch_size=20,
                    confidence_param=0.05,
                )
            
            def init_model(self, key):
                from flax import nnx
                return SimpleTestModel(key)
            
            def init_optimizer(self):
                import optax
                return optax.sgd(0.01, 0.9)
            
            def init_data(self, key):
                return x, y
            
            def loss_function(self, model_output, target):
                # Manual cross entropy implementation
                log_probs = jax.nn.log_softmax(model_output, axis=-1)
                one_hot_targets = jax.nn.one_hot(target, num_classes=model_output.shape[-1])
                return -jnp.mean(jnp.sum(log_probs * one_hot_targets, axis=-1))
            
            def accuracy(self, model_output, target):
                return jnp.mean(jnp.argmax(model_output, axis=-1) == target)
            
            def eval_p2l_convergence(self, model_output, target):
                # Manual cross entropy implementation
                log_probs = jax.nn.log_softmax(model_output, axis=-1)
                one_hot_targets = jax.nn.one_hot(target, num_classes=model_output.shape[-1])
                losses = -jnp.sum(log_probs * one_hot_targets, axis=-1)
                max_loss_index = jnp.argmax(losses)
                converged = losses[max_loss_index] <= 0.5
                return max_loss_index, converged
        
        config = SimpleTestConfig()
        
        # Run P2L
        start_time = time.time()
        result = pick_to_learn(config, self.key)
        end_time = time.time()
        
        # Validate results
        validation_results = {
            'success': True,
            'converged': result['converged'],
            'iterations': result['num_iterations'],
            'support_size': len(result['support_indices']),
            'generalization_bound': result['generalization_bound'],
            'training_time': end_time - start_time,
            'errors': []
        }
        
        # Check basic properties
        if result['num_iterations'] >= config.max_iterations:
            validation_results['errors'].append("P2L did not converge within max iterations")
        
        if len(result['support_indices']) == 0:
            validation_results['errors'].append("Support set is empty")
        
        if result['generalization_bound'] < 0 or result['generalization_bound'] > 1:
            validation_results['errors'].append("Generalization bound out of valid range [0,1]")
        
        if len(validation_results['errors']) > 0:
            validation_results['success'] = False
        
        print(f"Basic functionality test: {'PASS' if validation_results['success'] else 'FAIL'}")
        print(f"Converged: {validation_results['converged']}")
        print(f"Iterations: {validation_results['iterations']}")
        print(f"Support size: {validation_results['support_size']}")
        print(f"Generalization bound: {validation_results['generalization_bound']:.4f}")
        
        return validation_results
    
    def test_generalization_bound_calculation(self) -> Dict[str, Any]:
        """Test generalization bound calculation"""
        print("\n=== Testing Generalization Bound Calculation ===")
        
        # Test cases from the paper or known values
        test_cases = [
            {'k': 10, 'N': 100, 'beta': 0.05, 'expected_range': (0.1, 0.9)},
            {'k': 5, 'N': 50, 'beta': 0.1, 'expected_range': (0.05, 0.8)},
            {'k': 20, 'N': 200, 'beta': 0.035, 'expected_range': (0.15, 0.95)},
        ]
        
        validation_results = {
            'success': True,
            'test_cases': [],
            'errors': []
        }
        
        for i, case in enumerate(test_cases):
            bound = generalization_bound(case['k'], case['N'], case['beta'])
            
            test_result = {
                'case': i + 1,
                'k': case['k'],
                'N': case['N'],
                'beta': case['beta'],
                'computed_bound': bound,
                'expected_range': case['expected_range'],
                'in_range': case['expected_range'][0] <= bound <= case['expected_range'][1]
            }
            
            validation_results['test_cases'].append(test_result)
            
            if not test_result['in_range']:
                validation_results['errors'].append(
                    f"Case {i+1}: Bound {bound:.4f} not in expected range {case['expected_range']}"
                )
            
            print(f"Case {i+1}: k={case['k']}, N={case['N']}, β={case['beta']}")
            print(f"  Computed bound: {bound:.4f}")
            print(f"  Expected range: {case['expected_range']}")
            print(f"  In range: {test_result['in_range']}")
        
        if len(validation_results['errors']) > 0:
            validation_results['success'] = False
        
        print(f"Generalization bound test: {'PASS' if validation_results['success'] else 'FAIL'}")
        
        return validation_results
    
    def test_convergence_properties(self) -> Dict[str, Any]:
        """Test P2L convergence properties"""
        print("\n=== Testing P2L Convergence Properties ===")
        
        # Create multiple runs with different seeds
        n_runs = 10
        results = []
        
        for run in range(n_runs):
            key, self.key = jax.random.split(self.key)
            
            # Use a simple synthetic regression problem
            class ConvergenceTestConfig(SyntheticRegressionConfig):
                def __init__(self):
                    super().__init__(
                        n_datapoints=50,
                        noise_std=0.1,
                        learning_rate=0.01,
                        momentum=0.9,
                        convergence_param=0.2,
                        pretrain_fraction=0.4,
                        max_iterations=30,
                        train_epochs=20,
                        batch_size=25,
                        confidence_param=0.05,
                    )
            
            config = ConvergenceTestConfig()
            result = pick_to_learn(config, key)
            results.append(result)
        
        # Analyze convergence properties
        converged_count = sum(r['converged'] for r in results)
        support_sizes = [len(r['support_indices']) for r in results]
        iterations = [r['num_iterations'] for r in results]
        bounds = [r['generalization_bound'] for r in results]
        
        validation_results = {
            'success': True,
            'convergence_rate': converged_count / n_runs,
            'mean_support_size': np.mean(support_sizes),
            'std_support_size': np.std(support_sizes),
            'mean_iterations': np.mean(iterations),
            'std_iterations': np.std(iterations),
            'mean_bound': np.mean(bounds),
            'std_bound': np.std(bounds),
            'errors': []
        }
        
        # Check reasonable convergence rate (should be > 50% for simple problems)
        if validation_results['convergence_rate'] < 0.5:
            validation_results['errors'].append(
                f"Low convergence rate: {validation_results['convergence_rate']:.2f}"
            )
        
        # Check support size consistency
        if validation_results['std_support_size'] / validation_results['mean_support_size'] > 0.5:
            validation_results['errors'].append(
                f"High support size variance: {validation_results['std_support_size']:.2f}"
            )
        
        # Check bound consistency
        if validation_results['std_bound'] / validation_results['mean_bound'] > 0.5:
            validation_results['errors'].append(
                f"High bound variance: {validation_results['std_bound']:.4f}"
            )
        
        if len(validation_results['errors']) > 0:
            validation_results['success'] = False
        
        print(f"Convergence rate: {validation_results['convergence_rate']:.2f}")
        print(f"Mean support size: {validation_results['mean_support_size']:.1f} ± {validation_results['std_support_size']:.1f}")
        print(f"Mean iterations: {validation_results['mean_iterations']:.1f} ± {validation_results['std_iterations']:.1f}")
        print(f"Mean bound: {validation_results['mean_bound']:.4f} ± {validation_results['std_bound']:.4f}")
        print(f"Convergence properties test: {'PASS' if validation_results['success'] else 'FAIL'}")
        
        return validation_results
    
    def test_paper_replication_consistency(self) -> Dict[str, Any]:
        """Test consistency with paper results"""
        print("\n=== Testing Paper Replication Consistency ===")
        
        # Run experiments similar to the paper
        evaluator = P2LEvaluator(seed=self.seed)
        
        # Classification experiment
        class_config = BinaryMNISTP2LConfig(
            n_datapoints=500,  # Smaller for faster testing
            dataset_slice_index=0,
            dropout_prob=0.2,
            learning_rate=0.01,
            momentum=0.95,
            convergence_param=0.69314718,
            pretrain_fraction=0.5,
            max_iterations=100,  # Smaller for faster testing
            train_epochs=50,     # Smaller for faster testing
            batch_size=500,
            confidence_param=0.035,
        )
        
        # Regression experiment
        reg_config = SyntheticRegressionConfig(
            n_datapoints=100,  # Smaller for faster testing
            noise_std=0.1,
            learning_rate=0.01,
            momentum=0.9,
            convergence_param=0.1,
            pretrain_fraction=0.5,
            max_iterations=50,  # Smaller for faster testing
            train_epochs=30,    # Smaller for faster testing
            batch_size=100,
            confidence_param=0.035,
        )
        
        # Run experiments
        class_results = evaluator.run_classification_experiment(class_config, n_runs=3)
        reg_results = evaluator.run_regression_experiment(reg_config, n_runs=3)
        
        # Analyze results
        class_bounds = [r['generalization_bound'] for r in class_results]
        reg_bounds = [r['generalization_bound'] for r in reg_results]
        
        class_sizes = [len(r['support_indices']) for r in class_results]
        reg_sizes = [len(r['support_indices']) for r in reg_results]
        
        validation_results = {
            'success': True,
            'classification': {
                'mean_bound': np.mean(class_bounds),
                'std_bound': np.std(class_bounds),
                'mean_support_size': np.mean(class_sizes),
                'std_support_size': np.std(class_sizes),
            },
            'regression': {
                'mean_bound': np.mean(reg_bounds),
                'std_bound': np.std(reg_bounds),
                'mean_support_size': np.mean(reg_sizes),
                'std_support_size': np.std(reg_sizes),
            },
            'errors': []
        }
        
        # Check reasonable bounds (should be between 0.1 and 0.9 for these problems)
        if not (0.1 <= validation_results['classification']['mean_bound'] <= 0.9):
            validation_results['errors'].append(
                f"Classification bound out of reasonable range: {validation_results['classification']['mean_bound']:.4f}"
            )
        
        if not (0.1 <= validation_results['regression']['mean_bound'] <= 0.9):
            validation_results['errors'].append(
                f"Regression bound out of reasonable range: {validation_results['regression']['mean_bound']:.4f}"
            )
        
        # Check support sizes are reasonable (should be > 10% of dataset)
        if validation_results['classification']['mean_support_size'] < 50:
            validation_results['errors'].append(
                f"Classification support size too small: {validation_results['classification']['mean_support_size']:.1f}"
            )
        
        if validation_results['regression']['mean_support_size'] < 10:
            validation_results['errors'].append(
                f"Regression support size too small: {validation_results['regression']['mean_support_size']:.1f}"
            )
        
        if len(validation_results['errors']) > 0:
            validation_results['success'] = False
        
        print(f"Classification - Mean bound: {validation_results['classification']['mean_bound']:.4f} ± {validation_results['classification']['std_bound']:.4f}")
        print(f"Classification - Mean support size: {validation_results['classification']['mean_support_size']:.1f} ± {validation_results['classification']['std_support_size']:.1f}")
        print(f"Regression - Mean bound: {validation_results['regression']['mean_bound']:.4f} ± {validation_results['regression']['std_bound']:.4f}")
        print(f"Regression - Mean support size: {validation_results['regression']['mean_support_size']:.1f} ± {validation_results['regression']['std_support_size']:.1f}")
        print(f"Paper replication test: {'PASS' if validation_results['success'] else 'FAIL'}")
        
        return validation_results
    
    def run_full_validation(self) -> Dict[str, Any]:
        """Run all validation tests"""
        print("=== P2L Implementation Validation ===")
        
        results = {
            'basic_functionality': self.test_basic_functionality(),
            'generalization_bound': self.test_generalization_bound_calculation(),
            'convergence_properties': self.test_convergence_properties(),
            'paper_replication': self.test_paper_replication_consistency(),
        }
        
        # Overall assessment
        all_passed = all(r['success'] for r in results.values())
        total_errors = sum(len(r['errors']) for r in results.values())
        
        print(f"\n=== VALIDATION SUMMARY ===")
        print(f"Overall result: {'PASS' if all_passed else 'FAIL'}")
        print(f"Total errors: {total_errors}")
        
        for test_name, result in results.items():
            status = "PASS" if result['success'] else "FAIL"
            error_count = len(result['errors'])
            print(f"  {test_name}: {status} ({error_count} errors)")
        
        if not all_passed:
            print("\n=== DETAILED ERROR REPORT ===")
            for test_name, result in results.items():
                if result['errors']:
                    print(f"\n{test_name}:")
                    for error in result['errors']:
                        print(f"  - {error}")
        
        return {
            'overall_success': all_passed,
            'total_errors': total_errors,
            'test_results': results
        }


def main():
    """Run the full validation suite"""
    validator = P2LValidator(seed=42)
    results = validator.run_full_validation()
    return results


if __name__ == "__main__":
    main() 