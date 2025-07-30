"""Analysis tools for P2L implementation

This module provides detailed analysis of P2L behavior including:
1. Convergence analysis and visualization
2. Parameter sensitivity studies
3. Comparison with theoretical properties
4. Support set analysis
5. Generalization bound analysis
"""

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import pandas as pd

from p2l import P2LConfig, pick_to_learn, generalization_bound
from mnist_example import BinaryMNISTP2LConfig
from evaluation import SyntheticRegressionConfig, P2LEvaluator


@dataclass
class P2LAnalysisResult:
    """Container for P2L analysis results"""
    config: P2LConfig
    results: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class P2LAnalyzer:
    """Analyzer for P2L implementation"""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.key = jax.random.key(seed)
    
    def analyze_convergence_behavior(
        self, 
        config: P2LConfig, 
        n_runs: int = 10
    ) -> P2LAnalysisResult:
        """Analyze convergence behavior across multiple runs"""
        print(f"=== Analyzing Convergence Behavior ({n_runs} runs) ===")
        
        results = []
        convergence_curves = []
        
        for run in range(n_runs):
            print(f"Run {run + 1}/{n_runs}")
            key, self.key = jax.random.split(self.key)
            result = pick_to_learn(config, key)
            results.append(result)
            
            # Extract convergence curve (if available)
            if 'losses' in result:
                convergence_curves.append(result['losses'])
        
        # Analyze convergence patterns
        converged_count = sum(r['converged'] for r in results)
        iterations = [r['num_iterations'] for r in results]
        support_sizes = [len(r['support_indices']) for r in results]
        bounds = [r['generalization_bound'] for r in results]
        
        analysis_result = P2LAnalysisResult(
            config=config,
            results=results,
            metadata={
                'convergence_rate': converged_count / n_runs,
                'mean_iterations': np.mean(iterations),
                'std_iterations': np.std(iterations),
                'mean_support_size': np.mean(support_sizes),
                'std_support_size': np.std(support_sizes),
                'mean_bound': np.mean(bounds),
                'std_bound': np.std(bounds),
                'convergence_curves': convergence_curves,
            }
        )
        
        print(f"Convergence rate: {analysis_result.metadata['convergence_rate']:.2f}")
        print(f"Mean iterations: {analysis_result.metadata['mean_iterations']:.1f} ± {analysis_result.metadata['std_iterations']:.1f}")
        print(f"Mean support size: {analysis_result.metadata['mean_support_size']:.1f} ± {analysis_result.metadata['std_support_size']:.1f}")
        print(f"Mean bound: {analysis_result.metadata['mean_bound']:.4f} ± {analysis_result.metadata['std_bound']:.4f}")
        
        return analysis_result
    
    def analyze_parameter_sensitivity(
        self,
        base_config: P2LConfig,
        param_name: str,
        param_values: List[Any],
        n_runs_per_value: int = 5
    ) -> Dict[str, P2LAnalysisResult]:
        """Analyze sensitivity to a specific parameter"""
        print(f"=== Analyzing Parameter Sensitivity: {param_name} ===")
        
        sensitivity_results = {}
        
        for param_value in param_values:
            print(f"Testing {param_name} = {param_value}")
            
            # Create modified config
            modified_config = self._modify_config(base_config, param_name, param_value)
            
            # Run analysis
            analysis_result = self.analyze_convergence_behavior(
                modified_config, n_runs=n_runs_per_value
            )
            
            sensitivity_results[str(param_value)] = analysis_result
        
        return sensitivity_results
    
    def _modify_config(self, config: P2LConfig, param_name: str, param_value: Any) -> P2LConfig:
        """Create a modified config with a different parameter value"""
        # This is a simplified approach - in practice you'd need to handle different config types
        if hasattr(config, '__class__'):
            # Create a new instance of the same class
            config_dict = config.__dict__.copy()
            config_dict[param_name] = param_value
            return config.__class__(**config_dict)
        else:
            raise ValueError(f"Cannot modify config of type {type(config)}")
    
    def analyze_support_set_properties(
        self,
        analysis_result: P2LAnalysisResult
    ) -> Dict[str, Any]:
        """Analyze properties of the support sets"""
        print("=== Analyzing Support Set Properties ===")
        
        support_sets = [r['support_indices'] for r in analysis_result.results]
        
        # Analyze support set sizes
        sizes = [len(support_set) for support_set in support_sets]
        
        # Analyze overlap between support sets
        overlaps = []
        for i in range(len(support_sets)):
            for j in range(i + 1, len(support_sets)):
                overlap = len(set(support_sets[i]) & set(support_sets[j]))
                overlap_ratio = overlap / min(len(support_sets[i]), len(support_sets[j]))
                overlaps.append(overlap_ratio)
        
        # Analyze stability (how much support sets change between runs)
        stability_metrics = {
            'mean_size': np.mean(sizes),
            'std_size': np.std(sizes),
            'size_cv': np.std(sizes) / np.mean(sizes),  # Coefficient of variation
            'mean_overlap': np.mean(overlaps) if overlaps else 0,
            'std_overlap': np.std(overlaps) if overlaps else 0,
        }
        
        print(f"Support set size: {stability_metrics['mean_size']:.1f} ± {stability_metrics['std_size']:.1f}")
        print(f"Size coefficient of variation: {stability_metrics['size_cv']:.3f}")
        print(f"Mean overlap ratio: {stability_metrics['mean_overlap']:.3f} ± {stability_metrics['std_overlap']:.3f}")
        
        return stability_metrics
    
    def analyze_generalization_bounds(
        self,
        analysis_result: P2LAnalysisResult
    ) -> Dict[str, Any]:
        """Analyze generalization bound properties"""
        print("=== Analyzing Generalization Bounds ===")
        
        bounds = [r['generalization_bound'] for r in analysis_result.results]
        support_sizes = [len(r['support_indices']) for r in analysis_result.results]
        
        # Analyze bound properties
        bound_metrics = {
            'mean_bound': np.mean(bounds),
            'std_bound': np.std(bounds),
            'min_bound': np.min(bounds),
            'max_bound': np.max(bounds),
            'bound_cv': np.std(bounds) / np.mean(bounds),
        }
        
        # Analyze relationship between support size and bound
        correlation = np.corrcoef(support_sizes, bounds)[0, 1]
        bound_metrics['support_size_correlation'] = correlation
        
        print(f"Generalization bound: {bound_metrics['mean_bound']:.4f} ± {bound_metrics['std_bound']:.4f}")
        print(f"Bound range: [{bound_metrics['min_bound']:.4f}, {bound_metrics['max_bound']:.4f}]")
        print(f"Bound coefficient of variation: {bound_metrics['bound_cv']:.3f}")
        print(f"Correlation with support size: {correlation:.3f}")
        
        return bound_metrics
    
    def plot_convergence_analysis(
        self,
        analysis_result: P2LAnalysisResult,
        save_path: Optional[str] = None
    ):
        """Plot convergence analysis results"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Convergence curves
        if 'convergence_curves' in analysis_result.metadata:
            curves = analysis_result.metadata['convergence_curves']
            for i, curve in enumerate(curves[:5]):  # Plot first 5 curves
                axes[0, 0].plot(curve, alpha=0.7, label=f'Run {i+1}')
            axes[0, 0].set_xlabel('P2L Iteration')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('Convergence Curves')
            axes[0, 0].legend()
        
        # Plot 2: Support set size distribution
        support_sizes = [len(r['support_indices']) for r in analysis_result.results]
        axes[0, 1].hist(support_sizes, bins=10, alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Support Set Size')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Support Set Size Distribution')
        axes[0, 1].axvline(np.mean(support_sizes), color='red', linestyle='--',
                           label=f'Mean: {np.mean(support_sizes):.1f}')
        axes[0, 1].legend()
        
        # Plot 3: Generalization bound distribution
        bounds = [r['generalization_bound'] for r in analysis_result.results]
        axes[1, 0].hist(bounds, bins=10, alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('Generalization Bound')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Generalization Bound Distribution')
        axes[1, 0].axvline(np.mean(bounds), color='red', linestyle='--',
                           label=f'Mean: {np.mean(bounds):.4f}')
        axes[1, 0].legend()
        
        # Plot 4: Support size vs generalization bound
        axes[1, 1].scatter(support_sizes, bounds, alpha=0.7)
        axes[1, 1].set_xlabel('Support Set Size')
        axes[1, 1].set_ylabel('Generalization Bound')
        axes[1, 1].set_title('Support Size vs Generalization Bound')
        
        # Add trend line
        z = np.polyfit(support_sizes, bounds, 1)
        p = np.poly1d(z)
        axes[1, 1].plot(support_sizes, p(support_sizes), "r--", alpha=0.8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_parameter_sensitivity(
        self,
        sensitivity_results: Dict[str, P2LAnalysisResult],
        param_name: str,
        save_path: Optional[str] = None
    ):
        """Plot parameter sensitivity analysis"""
        
        param_values = [float(k) for k in sensitivity_results.keys()]
        convergence_rates = [r.metadata['convergence_rate'] for r in sensitivity_results.values()]
        mean_bounds = [r.metadata['mean_bound'] for r in sensitivity_results.values()]
        mean_support_sizes = [r.metadata['mean_support_size'] for r in sensitivity_results.values()]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot 1: Convergence rate
        axes[0].plot(param_values, convergence_rates, 'o-', linewidth=2, markersize=8)
        axes[0].set_xlabel(param_name)
        axes[0].set_ylabel('Convergence Rate')
        axes[0].set_title('Convergence Rate vs Parameter')
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Generalization bound
        axes[1].plot(param_values, mean_bounds, 'o-', linewidth=2, markersize=8)
        axes[1].set_xlabel(param_name)
        axes[1].set_ylabel('Mean Generalization Bound')
        axes[1].set_title('Generalization Bound vs Parameter')
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Support set size
        axes[2].plot(param_values, mean_support_sizes, 'o-', linewidth=2, markersize=8)
        axes[2].set_xlabel(param_name)
        axes[2].set_ylabel('Mean Support Set Size')
        axes[2].set_title('Support Set Size vs Parameter')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def compare_with_theoretical_properties(
        self,
        analysis_result: P2LAnalysisResult
    ) -> Dict[str, Any]:
        """Compare empirical results with theoretical properties"""
        print("=== Comparing with Theoretical Properties ===")
        
        # Extract metrics
        convergence_rate = analysis_result.metadata['convergence_rate']
        mean_support_size = analysis_result.metadata['mean_support_size']
        mean_bound = analysis_result.metadata['mean_bound']
        
        # Theoretical expectations (based on paper)
        theoretical_analysis = {
            'convergence_rate_expected': 0.8,  # Should converge in most cases
            'support_size_ratio_expected': 0.3,  # Support set should be smaller than full dataset
            'bound_range_expected': (0.1, 0.9),  # Reasonable bound range
        }
        
        # Compare with expectations
        comparisons = {
            'convergence_rate_ok': convergence_rate >= 0.5,  # Should converge in at least 50% of cases
            'support_size_ok': mean_support_size > 0,  # Should have non-empty support set
            'bound_in_range': theoretical_analysis['bound_range_expected'][0] <= mean_bound <= theoretical_analysis['bound_range_expected'][1],
        }
        
        print(f"Convergence rate: {convergence_rate:.2f} (expected > 0.5)")
        print(f"Mean support size: {mean_support_size:.1f} (expected > 0)")
        print(f"Mean bound: {mean_bound:.4f} (expected in {theoretical_analysis['bound_range_expected']})")
        
        all_ok = all(comparisons.values())
        print(f"Theoretical consistency: {'PASS' if all_ok else 'FAIL'}")
        
        return {
            'comparisons': comparisons,
            'theoretical_expectations': theoretical_analysis,
            'empirical_results': {
                'convergence_rate': convergence_rate,
                'mean_support_size': mean_support_size,
                'mean_bound': mean_bound,
            }
        }


def run_comprehensive_analysis():
    """Run comprehensive analysis of P2L implementation"""
    
    print("=== Comprehensive P2L Analysis ===")
    
    analyzer = P2LAnalyzer(seed=42)
    
    # Create test configurations
    class_config = BinaryMNISTP2LConfig(
        n_datapoints=500,
        dataset_slice_index=0,
        dropout_prob=0.2,
        learning_rate=0.01,
        momentum=0.95,
        convergence_param=0.69314718,
        pretrain_fraction=0.5,
        max_iterations=100,
        train_epochs=50,
        batch_size=500,
        confidence_param=0.035,
    )
    
    reg_config = SyntheticRegressionConfig(
        n_datapoints=100,
        noise_std=0.1,
        learning_rate=0.01,
        momentum=0.9,
        convergence_param=0.1,
        pretrain_fraction=0.5,
        max_iterations=50,
        train_epochs=30,
        batch_size=100,
        confidence_param=0.035,
    )
    
    # Analyze classification
    print("\n--- Classification Analysis ---")
    class_analysis = analyzer.analyze_convergence_behavior(class_config, n_runs=5)
    analyzer.analyze_support_set_properties(class_analysis)
    analyzer.analyze_generalization_bounds(class_analysis)
    analyzer.plot_convergence_analysis(class_analysis, save_path="./results/classification_analysis.png")
    analyzer.compare_with_theoretical_properties(class_analysis)
    
    # Analyze regression
    print("\n--- Regression Analysis ---")
    reg_analysis = analyzer.analyze_convergence_behavior(reg_config, n_runs=5)
    analyzer.analyze_support_set_properties(reg_analysis)
    analyzer.analyze_generalization_bounds(reg_analysis)
    analyzer.plot_convergence_analysis(reg_analysis, save_path="./results/regression_analysis.png")
    analyzer.compare_with_theoretical_properties(reg_analysis)
    
    # Parameter sensitivity analysis
    print("\n--- Parameter Sensitivity Analysis ---")
    learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1]
    lr_sensitivity = analyzer.analyze_parameter_sensitivity(
        class_config, 'learning_rate', learning_rates, n_runs_per_value=3
    )
    analyzer.plot_parameter_sensitivity(
        lr_sensitivity, 'learning_rate', save_path="./results/learning_rate_sensitivity.png"
    )
    
    return {
        'classification_analysis': class_analysis,
        'regression_analysis': reg_analysis,
        'learning_rate_sensitivity': lr_sensitivity,
    }


if __name__ == "__main__":
    run_comprehensive_analysis() 