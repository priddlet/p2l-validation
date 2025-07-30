"""Main evaluation script for P2L implementation

This script provides a comprehensive evaluation framework that:
1. Validates the P2L implementation
2. Replicates paper results
3. Analyzes P2L behavior
4. Generates comparison plots
5. Provides insights for tuning

Usage:
    python run_evaluation.py --mode validation
    python run_evaluation.py --mode replication
    python run_evaluation.py --mode analysis
    python run_evaluation.py --mode all
"""

import argparse
import os
import sys
from pathlib import Path
import time
from typing import Dict, Any

# Add the current directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from validation import P2LValidator
from evaluation import run_paper_replication, P2LEvaluator
from analysis import P2LAnalyzer, run_comprehensive_analysis


def run_validation_mode():
    """Run validation tests"""
    print("=" * 60)
    print("P2L IMPLEMENTATION VALIDATION")
    print("=" * 60)
    
    validator = P2LValidator(seed=42)
    results = validator.run_full_validation()
    
    return results


def run_replication_mode():
    """Run paper replication experiments"""
    print("=" * 60)
    print("P2L PAPER REPLICATION")
    print("=" * 60)
    
    # Create results directory
    results_dir = Path("./results")
    results_dir.mkdir(exist_ok=True)
    
    # Run replication
    class_results, reg_results = run_paper_replication()
    
    return {
        'classification_results': class_results,
        'regression_results': reg_results,
    }


def run_analysis_mode():
    """Run comprehensive analysis"""
    print("=" * 60)
    print("P2L COMPREHENSIVE ANALYSIS")
    print("=" * 60)
    
    # Create results directory
    results_dir = Path("./results")
    results_dir.mkdir(exist_ok=True)
    
    # Run analysis
    analysis_results = run_comprehensive_analysis()
    
    return analysis_results


def run_all_modes():
    """Run all evaluation modes"""
    print("=" * 60)
    print("COMPREHENSIVE P2L EVALUATION")
    print("=" * 60)
    
    all_results = {}
    
    # 1. Validation
    print("\n" + "="*40)
    print("STEP 1: VALIDATION")
    print("="*40)
    validation_results = run_validation_mode()
    all_results['validation'] = validation_results
    
    # 2. Replication
    print("\n" + "="*40)
    print("STEP 2: PAPER REPLICATION")
    print("="*40)
    replication_results = run_replication_mode()
    all_results['replication'] = replication_results
    
    # 3. Analysis
    print("\n" + "="*40)
    print("STEP 3: COMPREHENSIVE ANALYSIS")
    print("="*40)
    analysis_results = run_analysis_mode()
    all_results['analysis'] = analysis_results
    
    return all_results


def generate_summary_report(results: Dict[str, Any]):
    """Generate a summary report of all results"""
    print("\n" + "="*60)
    print("EVALUATION SUMMARY REPORT")
    print("="*60)
    
    # Validation summary
    if 'validation' in results:
        validation = results['validation']
        print(f"\nVALIDATION RESULTS:")
        print(f"  Overall success: {'PASS' if validation['overall_success'] else 'FAIL'}")
        print(f"  Total errors: {validation['total_errors']}")
        
        for test_name, result in validation['test_results'].items():
            status = "PASS" if result['success'] else "FAIL"
            error_count = len(result['errors'])
            print(f"  {test_name}: {status} ({error_count} errors)")
    
    # Replication summary
    if 'replication' in results:
        replication = results['replication']
        print(f"\nREPLICATION RESULTS:")
        
        if 'classification_results' in replication:
            class_bounds = [r['generalization_bound'] for r in replication['classification_results']]
            class_sizes = [len(r['support_indices']) for r in replication['classification_results']]
            print(f"  Classification - Mean bound: {np.mean(class_bounds):.4f} ± {np.std(class_bounds):.4f}")
            print(f"  Classification - Mean support size: {np.mean(class_sizes):.1f} ± {np.std(class_sizes):.1f}")
        
        if 'regression_results' in replication:
            reg_bounds = [r['generalization_bound'] for r in replication['regression_results']]
            reg_sizes = [len(r['support_indices']) for r in replication['regression_results']]
            print(f"  Regression - Mean bound: {np.mean(reg_bounds):.4f} ± {np.std(reg_bounds):.4f}")
            print(f"  Regression - Mean support size: {np.mean(reg_sizes):.1f} ± {np.std(reg_sizes):.1f}")
    
    # Analysis summary
    if 'analysis' in results:
        analysis = results['analysis']
        print(f"\nANALYSIS RESULTS:")
        
        if 'classification_analysis' in analysis:
            class_meta = analysis['classification_analysis'].metadata
            print(f"  Classification convergence rate: {class_meta['convergence_rate']:.2f}")
            print(f"  Classification mean bound: {class_meta['mean_bound']:.4f} ± {class_meta['std_bound']:.4f}")
        
        if 'regression_analysis' in analysis:
            reg_meta = analysis['regression_analysis'].metadata
            print(f"  Regression convergence rate: {reg_meta['convergence_rate']:.2f}")
            print(f"  Regression mean bound: {reg_meta['mean_bound']:.4f} ± {reg_meta['std_bound']:.4f}")
    
    # Overall assessment
    print(f"\nOVERALL ASSESSMENT:")
    
    validation_ok = results.get('validation', {}).get('overall_success', False)
    replication_ok = 'replication' in results and len(results['replication']) > 0
    analysis_ok = 'analysis' in results and len(results['analysis']) > 0
    
    print(f"  Validation: {'PASS' if validation_ok else 'FAIL'}")
    print(f"  Replication: {'PASS' if replication_ok else 'FAIL'}")
    print(f"  Analysis: {'PASS' if analysis_ok else 'FAIL'}")
    
    overall_success = validation_ok and replication_ok and analysis_ok
    print(f"  Overall: {'PASS' if overall_success else 'FAIL'}")
    
    # Recommendations
    print(f"\nRECOMMENDATIONS:")
    if validation_ok:
        print("  ✓ P2L implementation appears to be working correctly")
    else:
        print("  ✗ P2L implementation has issues that need to be addressed")
    
    if replication_ok:
        print("  ✓ Paper replication successful")
    else:
        print("  ✗ Paper replication failed or incomplete")
    
    if analysis_ok:
        print("  ✓ Comprehensive analysis completed")
        print("  ✓ Check ./results/ directory for detailed plots and analysis")
    else:
        print("  ✗ Analysis incomplete")
    
    # Analysis summary
    if 'analysis' in results:
        analysis = results['analysis']
        print(f"\nANALYSIS RESULTS:")
        
        if 'classification_analysis' in analysis:
            class_meta = analysis['classification_analysis'].metadata
            print(f"  Classification convergence rate: {class_meta['convergence_rate']:.2f}")
            print(f"  Classification mean bound: {class_meta['mean_bound']:.4f} ± {class_meta['std_bound']:.4f}")
        
        if 'regression_analysis' in analysis:
            reg_meta = analysis['regression_analysis'].metadata
            print(f"  Regression convergence rate: {reg_meta['convergence_rate']:.2f}")
            print(f"  Regression mean bound: {reg_meta['mean_bound']:.4f} ± {reg_meta['std_bound']:.4f}")
    
    # Overall assessment
    print(f"\nOVERALL ASSESSMENT:")
    
    validation_ok = results.get('validation', {}).get('overall_success', False)
    replication_ok = 'replication' in results and len(results['replication']) > 0
    analysis_ok = 'analysis' in results and len(results['analysis']) > 0
    
    print(f"  Validation: {'PASS' if validation_ok else 'FAIL'}")
    print(f"  Replication: {'PASS' if replication_ok else 'FAIL'}")
    print(f"  Analysis: {'PASS' if analysis_ok else 'FAIL'}")
    
    overall_success = validation_ok and replication_ok and analysis_ok
    print(f"  Overall: {'PASS' if overall_success else 'FAIL'}")
    
    # Recommendations
    print(f"\nRECOMMENDATIONS:")
    if validation_ok:
        print("  ✓ P2L implementation appears to be working correctly")
    else:
        print("  ✗ P2L implementation has issues that need to be addressed")
    
    if replication_ok:
        print("  ✓ Paper replication successful")
    else:
        print("  ✗ Paper replication failed or incomplete")
    
    if analysis_ok:
        print("  ✓ Comprehensive analysis completed")
        print("  ✓ Check ./results/ directory for detailed plots and analysis")
    else:
        print("  ✗ Analysis incomplete")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="P2L Implementation Evaluation")
    parser.add_argument(
        "--mode", 
        choices=["validation", "replication", "analysis", "all"],
        default="all",
        help="Evaluation mode to run"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Set random seed
    import numpy as np
    import jax
    np.random.seed(args.seed)
    jax.random.key(args.seed)
    
    # Create results directory
    results_dir = Path("./results")
    results_dir.mkdir(exist_ok=True)
    
    start_time = time.time()
    
    try:
        if args.mode == "validation":
            results = run_validation_mode()
        elif args.mode == "replication":
            results = run_replication_mode()
        elif args.mode == "analysis":
            results = run_analysis_mode()
        elif args.mode == "all":
            results = run_all_modes()
        else:
            raise ValueError(f"Unknown mode: {args.mode}")
        
        # Generate summary report
        generate_summary_report(results)
        
        end_time = time.time()
        print(f"\nTotal evaluation time: {end_time - start_time:.2f} seconds")
        
        # Save results
        import pickle
        with open(results_dir / "evaluation_results.pkl", "wb") as f:
            pickle.dump(results, f)
        
        print(f"\nResults saved to: {results_dir}")
        
    except Exception as e:
        print(f"Evaluation failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 