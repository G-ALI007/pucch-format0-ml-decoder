"""
================================================================================
PUCCH Format 0 - Neural Network Architecture Comparison
================================================================================

This script compares different neural network architectures for
PUCCH Format 0 decoding and analyzes computational complexity.

Architectures compared:
    1. FC-Small:  [32, 32]     - 2 layers, 32 neurons each
    2. FC-Medium: [128, 128]   - 2 layers, 128 neurons each (paper's choice)
    3. FC-Large:  [256, 256]   - 2 layers, 256 neurons each
    4. FC-Deep:   [128, 128, 128] - 3 layers, 128 neurons each

Complexity analysis:
    - Number of parameters
    - Number of multiply operations
    - Inference time comparison
    - Comparison with 12-point DFT

Usage:
    python main_architectures.py
    python main_architectures.py --no-plots

================================================================================
"""

from evaluation import evaluate_all_snr
from model import (
    set_random_seeds,
    create_model,
    load_saved_model,
    predict,
    print_tensorflow_info
)
from data_preprocessing import prepare_data, preprocess_pipeline
from data_loader import load_all_datasets, analyze_dataset
from config import config
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from tf_keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    CSVLogger
)
from tf_keras.utils import to_categorical
import os
import sys
import time
import datetime
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional

# Fix for TensorFlow >= 2.16
os.environ['TF_USE_LEGACY_KERAS'] = '1'

# tf_keras imports

# Scikit-learn

# Visualization

# Import project modules

# =============================================================================
# SECTION 1: CONFIGURATION
# =============================================================================

# Output directories
ARCH_RESULTS_DIR = "./results_architectures/"
ARCH_MODELS_DIR = "./models_architectures/"
ARCH_PLOTS_DIR = "./plots_architectures/"
ARCH_LOGS_DIR = "./logs_architectures/"

# Architectures to compare
ARCHITECTURES = {
    'FC_Small_2x32': {
        'name': '2 Layers × 32',
        'hidden_layers': [32, 32],
        'short_name': '2×32'
    },
    'FC_Medium_2x128': {
        'name': '2 Layers × 128 (Paper)',
        'hidden_layers': [128, 128],
        'short_name': '2×128'
    },
    'FC_Large_2x256': {
        'name': '2 Layers × 256',
        'hidden_layers': [256, 256],
        'short_name': '2×256'
    },
    'FC_Deep_3x128': {
        'name': '3 Layers × 128',
        'hidden_layers': [128, 128, 128],
        'short_name': '3×128'
    }
}

# DFT complexity (12-point DFT for comparison)
DFT_12_POINT_MULTIPLICATIONS = 12 * 12  # 144 complex multiplications
DFT_12_POINT_ADDITIONS = 12 * 11        # 132 complex additions


def create_arch_directories():
    """Create output directories."""
    for d in [ARCH_RESULTS_DIR, ARCH_MODELS_DIR, ARCH_PLOTS_DIR, ARCH_LOGS_DIR]:
        if d and not os.path.exists(d):
            os.makedirs(d)
            print(f"Created: {d}")


# =============================================================================
# SECTION 2: COMPLEXITY ANALYSIS
# =============================================================================

def compute_model_complexity(hidden_layers: List[int]) -> Dict:
    """
    Compute computational complexity of a fully connected neural network.

    For a Dense layer with M inputs and N outputs:
        Multiplications = M × N (weights) + N (bias) ≈ M × N
        Additions = M × N (accumulation) + N (bias) ≈ M × N
        Parameters = M × N + N (weights + biases)

    Parameters:
    -----------
    hidden_layers : list of int
        Hidden layer sizes (e.g., [128, 128])

    Returns:
    --------
    complexity : dict
        Dictionary containing:
        - total_params: Total number of parameters
        - total_multiplications: Total multiply operations
        - total_additions: Total addition operations
        - total_flops: Total floating point operations
        - layer_details: Per-layer breakdown
        - dft_ratio: Ratio compared to 12-point DFT
    """

    input_size = config.INPUT_SIZE    # 24
    output_size = config.OUTPUT_SIZE  # 4

    # Build layer sizes: [input] + hidden + [output]
    layer_sizes = [input_size] + hidden_layers + [output_size]

    total_params = 0
    total_multiplications = 0
    total_additions = 0
    layer_details = []

    for i in range(len(layer_sizes) - 1):
        m = layer_sizes[i]       # Input neurons
        n = layer_sizes[i + 1]   # Output neurons

        # Parameters: weights (M×N) + biases (N)
        params = m * n + n

        # Multiplications: M×N (matrix multiply)
        multiplications = m * n

        # Additions: (M-1)×N (accumulation) + N (bias)
        additions = (m - 1) * n + n

        total_params += params
        total_multiplications += multiplications
        total_additions += additions

        layer_name = f"Layer {i+1}"
        if i < len(hidden_layers):
            layer_name = f"Hidden {i+1} ({n} neurons)"
        else:
            layer_name = f"Output ({n} neurons)"

        layer_details.append({
            'name': layer_name,
            'input_size': m,
            'output_size': n,
            'params': params,
            'multiplications': multiplications,
            'additions': additions
        })

    # Total FLOPs (multiply + add = 2 operations each)
    total_flops = total_multiplications + total_additions

    # Comparison with DFT
    dft_mult_ratio = total_multiplications / DFT_12_POINT_MULTIPLICATIONS

    complexity = {
        'hidden_layers': hidden_layers,
        'total_params': total_params,
        'total_multiplications': total_multiplications,
        'total_additions': total_additions,
        'total_flops': total_flops,
        'layer_details': layer_details,
        'dft_multiplication_ratio': dft_mult_ratio,
        'dft_12_point_multiplications': DFT_12_POINT_MULTIPLICATIONS
    }

    return complexity


def analyze_all_complexities(verbose: bool = True) -> Dict:
    """
    Analyze complexity for all architectures.

    Returns:
    --------
    all_complexity : dict mapping architecture name to complexity dict
    """

    if verbose:
        print("\n" + "=" * 70)
        print("COMPUTATIONAL COMPLEXITY ANALYSIS")
        print("=" * 70)

    all_complexity = {}

    for arch_key, arch_info in ARCHITECTURES.items():
        complexity = compute_model_complexity(arch_info['hidden_layers'])
        all_complexity[arch_key] = complexity
        all_complexity[arch_key]['name'] = arch_info['name']
        all_complexity[arch_key]['short_name'] = arch_info['short_name']

    if verbose:
        # Print comparison table
        print(f"\n{'Architecture':<25}{'Params':<12}{'Multiply':<12}"
              f"{'Add':<12}{'FLOPs':<12}{'vs DFT':<10}")
        print("-" * 83)

        # DFT baseline
        print(f"{'12-point DFT':<25}{'-':<12}{DFT_12_POINT_MULTIPLICATIONS:<12}"
              f"{DFT_12_POINT_ADDITIONS:<12}"
              f"{DFT_12_POINT_MULTIPLICATIONS + DFT_12_POINT_ADDITIONS:<12}{'1.0x':<10}")

        print("-" * 83)

        for arch_key in ARCHITECTURES.keys():
            c = all_complexity[arch_key]
            print(f"{c['name']:<25}"
                  f"{c['total_params']:<12,}"
                  f"{c['total_multiplications']:<12,}"
                  f"{c['total_additions']:<12,}"
                  f"{c['total_flops']:<12,}"
                  f"{c['dft_multiplication_ratio']:<10.1f}x")

        print("-" * 83)

        # Print layer-by-layer for paper's architecture
        print(f"\nDetailed breakdown for 2×128 (paper's architecture):")
        print(
            f"{'Layer':<25}{'In':<8}{'Out':<8}{'Params':<10}{'Multiply':<10}{'Add':<10}")
        print("-" * 71)

        for layer in all_complexity['FC_Medium_2x128']['layer_details']:
            print(f"{layer['name']:<25}"
                  f"{layer['input_size']:<8}"
                  f"{layer['output_size']:<8}"
                  f"{layer['params']:<10,}"
                  f"{layer['multiplications']:<10,}"
                  f"{layer['additions']:<10,}")

        print("-" * 71)
        print(f"{'Total':<25}{'':<8}{'':<8}"
              f"{all_complexity['FC_Medium_2x128']['total_params']:<10,}"
              f"{all_complexity['FC_Medium_2x128']['total_multiplications']:<10,}"
              f"{all_complexity['FC_Medium_2x128']['total_additions']:<10,}")

    if verbose:
        print("=" * 70 + "\n")

    return all_complexity


# =============================================================================
# SECTION 3: TRAIN AND EVALUATE ALL ARCHITECTURES
# =============================================================================

def train_single_architecture(
    arch_key: str,
    arch_info: Dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    verbose: int = 1
) -> Tuple[object, Dict, float]:
    """
    Train a single architecture.

    Returns:
    --------
    model : trained model
    history : training history dict
    training_time : seconds
    """

    print(f"\n{'='*70}")
    print(f"Training: {arch_info['name']}")
    print(f"Hidden layers: {arch_info['hidden_layers']}")
    print(f"{'='*70}")

    # Create model
    model = create_model(
        input_size=config.INPUT_SIZE,
        hidden_layers=arch_info['hidden_layers'],
        output_size=config.OUTPUT_SIZE,
        hidden_activation=config.HIDDEN_ACTIVATION,
        output_activation=config.OUTPUT_ACTIVATION,
        dropout_rate=config.DROPOUT_RATE,
        use_dropout=config.USE_DROPOUT,
        kernel_initializer=config.KERNEL_INITIALIZER,
        learning_rate=config.LEARNING_RATE,
        momentum=config.MOMENTUM,
        use_nesterov=config.USE_NESTEROV,
        print_summary=True
    )

    # One-hot encode labels
    y_train_oh = to_categorical(y_train, num_classes=config.NUM_CLASSES)
    y_val_oh = to_categorical(y_val, num_classes=config.NUM_CLASSES)

    # Paths
    model_path = os.path.join(ARCH_MODELS_DIR, f"{arch_key}.h5")
    history_path = os.path.join(ARCH_LOGS_DIR, f"{arch_key}_history.csv")

    for path in [model_path, history_path]:
        d = os.path.dirname(path)
        if d and not os.path.exists(d):
            os.makedirs(d)

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            filepath=model_path,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            mode='min',
            patience=config.EARLY_STOPPING_PATIENCE,
            min_delta=config.EARLY_STOPPING_MIN_DELTA,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            mode='min',
            factor=config.REDUCE_LR_FACTOR,
            patience=config.REDUCE_LR_PATIENCE,
            min_lr=config.REDUCE_LR_MIN_LR,
            verbose=1
        ),
        CSVLogger(
            filename=history_path,
            separator=',',
            append=False
        )
    ]

    # Train
    print(f"\nTraining {arch_info['name']}...")
    print("-" * 70)

    start_time = time.time()

    history_obj = model.fit(
        x=X_train,
        y=y_train_oh,
        validation_data=(X_val, y_val_oh),
        epochs=config.NUM_EPOCHS,
        batch_size=config.BATCH_SIZE,
        callbacks=callbacks,
        verbose=verbose
    )

    training_time = time.time() - start_time
    history = history_obj.history

    print("-" * 70)
    print(f"Training complete: {training_time:.1f}s")
    print(f"Best val acc: {max(history['val_accuracy'])*100:.2f}%")

    # Load best model
    try:
        best_model = load_saved_model(model_path)
    except FileNotFoundError:
        best_model = model

    return best_model, history, training_time


def measure_inference_time(
    model,
    X: np.ndarray,
    num_runs: int = 10,
    verbose: bool = True
) -> Dict:
    """
    Measure inference time for a model.

    Parameters:
    -----------
    model : keras model
    X : np.ndarray
        Test data
    num_runs : int
        Number of measurement runs
    verbose : bool
        Print results

    Returns:
    --------
    timing : dict
        - total_time: Total time for all runs
        - avg_time: Average time per run
        - avg_per_sample: Average time per sample
        - samples_per_second: Throughput
    """

    num_samples = len(X)
    times = []

    # Warm-up run
    _ = model.predict(X[:100], verbose=0)

    # Timed runs
    for run in range(num_runs):
        start = time.time()
        _ = model.predict(X, batch_size=256, verbose=0)
        elapsed = time.time() - start
        times.append(elapsed)

    avg_time = float(np.mean(times))
    avg_per_sample = avg_time / num_samples
    samples_per_second = num_samples / avg_time

    timing = {
        'num_samples': num_samples,
        'num_runs': num_runs,
        'total_time': float(sum(times)),
        'avg_time': avg_time,
        'avg_per_sample_ms': avg_per_sample * 1000,
        'samples_per_second': samples_per_second
    }

    if verbose:
        print(f"  Inference: {avg_time*1000:.2f}ms for {num_samples:,} samples "
              f"({avg_per_sample*1000:.4f}ms/sample, {samples_per_second:,.0f} samples/s)")

    return timing


# =============================================================================
# SECTION 4: VISUALIZATION
# =============================================================================

def plot_architecture_comparison(
    arch_results: Dict,
    save_path: Optional[str] = None,
    show_plot: bool = True
):
    """
    Plot accuracy vs SNR for all architectures.
    """

    if save_path is None:
        save_path = os.path.join(ARCH_PLOTS_DIR, "architecture_comparison.png")

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['blue', 'red', 'green', 'orange']
    markers = ['o', 's', '^', 'D']

    for idx, (arch_key, results) in enumerate(arch_results.items()):
        snr_values = sorted(results['snr_accuracy'].keys())
        accuracies = [results['snr_accuracy'][snr] * 100 for snr in snr_values]

        arch_name = ARCHITECTURES[arch_key]['short_name']

        ax.plot(snr_values, accuracies,
                color=colors[idx % len(colors)],
                marker=markers[idx % len(markers)],
                linewidth=2.5, markersize=10,
                label=arch_name,
                markerfacecolor='white', markeredgewidth=2)

    ax.axhline(y=99, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.set_xlabel('SNR (dB)', fontsize=14)
    ax.set_ylabel('Accuracy (%)', fontsize=14)
    ax.set_title(
        'Neural Network Performance for Various Architectures', fontsize=16)
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(snr_values)

    all_acc = []
    for results in arch_results.values():
        for snr in results['snr_accuracy']:
            all_acc.append(results['snr_accuracy'][snr] * 100)
    ax.set_ylim([max(0, min(all_acc) - 5), 102])

    plt.tight_layout()

    d = os.path.dirname(save_path)
    if d and not os.path.exists(d):
        os.makedirs(d)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def plot_complexity_vs_accuracy(
    arch_results: Dict,
    all_complexity: Dict,
    save_path: Optional[str] = None,
    show_plot: bool = True
):
    """
    Plot complexity (parameters) vs accuracy for all architectures.
    """

    if save_path is None:
        save_path = os.path.join(ARCH_PLOTS_DIR, "complexity_vs_accuracy.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    arch_names = []
    params = []
    accuracies_10db = []
    inference_times = []

    for arch_key in ARCHITECTURES.keys():
        if arch_key in arch_results:
            arch_names.append(ARCHITECTURES[arch_key]['short_name'])
            params.append(all_complexity[arch_key]['total_params'])
            accuracies_10db.append(
                arch_results[arch_key]['snr_accuracy'].get(10, 0) * 100)
            inference_times.append(
                arch_results[arch_key]['inference_time']['avg_per_sample_ms'])

    # Left: Parameters vs Accuracy
    colors = ['blue', 'red', 'green', 'orange']

    for i, (name, p, acc) in enumerate(zip(arch_names, params, accuracies_10db)):
        axes[0].scatter(p, acc, color=colors[i], s=200, zorder=5,
                        edgecolors='black', linewidths=1.5)
        axes[0].annotate(name, (p, acc), textcoords="offset points",
                         xytext=(10, 5), fontsize=11, fontweight='bold')

    axes[0].set_xlabel('Number of Parameters', fontsize=14)
    axes[0].set_ylabel('Accuracy at SNR=10 dB (%)', fontsize=14)
    axes[0].set_title('Parameters vs Accuracy', fontsize=16)
    axes[0].grid(True, alpha=0.3)

    # Right: Parameters vs Inference Time
    for i, (name, p, t) in enumerate(zip(arch_names, params, inference_times)):
        axes[1].scatter(p, t, color=colors[i], s=200, zorder=5,
                        edgecolors='black', linewidths=1.5)
        axes[1].annotate(name, (p, t), textcoords="offset points",
                         xytext=(10, 5), fontsize=11, fontweight='bold')

    axes[1].set_xlabel('Number of Parameters', fontsize=14)
    axes[1].set_ylabel('Inference Time (ms/sample)', fontsize=14)
    axes[1].set_title('Parameters vs Inference Time', fontsize=16)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    d = os.path.dirname(save_path)
    if d and not os.path.exists(d):
        os.makedirs(d)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


# =============================================================================
# SECTION 5: SAVE RESULTS
# =============================================================================

def save_architecture_results(
    arch_results: Dict,
    all_complexity: Dict,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Save architecture comparison results to CSV.
    """

    filepath = os.path.join(ARCH_RESULTS_DIR, "architecture_comparison.csv")

    d = os.path.dirname(filepath)
    if d and not os.path.exists(d):
        os.makedirs(d)

    data = []

    for arch_key in ARCHITECTURES.keys():
        if arch_key not in arch_results:
            continue

        results = arch_results[arch_key]
        complexity = all_complexity[arch_key]

        row = {
            'Architecture': ARCHITECTURES[arch_key]['name'],
            'Hidden_Layers': str(ARCHITECTURES[arch_key]['hidden_layers']),
            'Parameters': complexity['total_params'],
            'Multiplications': complexity['total_multiplications'],
            'FLOPs': complexity['total_flops'],
            'DFT_Ratio': complexity['dft_multiplication_ratio'],
            'Training_Time_s': results['training_time'],
            'Inference_ms_per_sample': results['inference_time']['avg_per_sample_ms'],
            'Best_Val_Accuracy': results['best_val_accuracy']
        }

        # Per-SNR accuracy
        for snr in sorted(results['snr_accuracy'].keys()):
            row[f'Accuracy_SNR_{snr}dB'] = results['snr_accuracy'][snr] * 100

        data.append(row)

    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False, float_format='%.4f')

    if verbose:
        print(f"\nResults saved to: {filepath}")
        print("\nArchitecture Comparison:")
        print("-" * 90)
        display_cols = ['Architecture', 'Parameters', 'DFT_Ratio',
                        'Inference_ms_per_sample', 'Best_Val_Accuracy']
        for snr in config.SNR_VALUES:
            col = f'Accuracy_SNR_{snr}dB'
            if col in df.columns:
                display_cols.append(col)
        print(df[display_cols].to_string(index=False))
        print("-" * 90)

    return df


# =============================================================================
# SECTION 6: MAIN PIPELINE
# =============================================================================

def run_architecture_comparison(show_plots: bool = True):
    """
    Run the complete architecture comparison pipeline.
    """

    pipeline_start = time.time()

    print("\n" + "#" * 70)
    print("#" + " " * 10 + "ARCHITECTURE COMPARISON PIPELINE" + " " * 27 + "#")
    print("#" * 70)

    print(
        f"\nTimestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print_tensorflow_info()

    create_arch_directories()
    set_random_seeds(config.MASTER_SEED)

    # =========================================================================
    # STEP 1: Complexity Analysis
    # =========================================================================

    print("\n" + "=" * 70)
    print("STEP 1: Computational Complexity Analysis")
    print("=" * 70)

    all_complexity = analyze_all_complexities(verbose=True)

    # =========================================================================
    # STEP 2: Load Data
    # =========================================================================

    print("\n" + "=" * 70)
    print("STEP 2: Loading Data")
    print("=" * 70)

    try:
        datasets = load_all_datasets(verbose=True)
    except Exception as e:
        print(f"ERROR: {e}")
        return None

    # Prepare data
    X_train, y_train, X_val, y_val, X_test_dict, y_test_dict = prepare_data(
        datasets=datasets,
        train_snr=config.TRAIN_SNR,
        verbose=True
    )

    # Preprocess
    X_train_proc, X_val_proc, X_test_dict_proc, scaler = preprocess_pipeline(
        X_train=X_train,
        X_val=X_val,
        X_test_dict=X_test_dict,
        normalize=config.NORMALIZE_FEATURES,
        verbose=True
    )

    # =========================================================================
    # STEP 3: Train and Evaluate All Architectures
    # =========================================================================

    print("\n" + "=" * 70)
    print("STEP 3: Training and Evaluating All Architectures")
    print("=" * 70)

    arch_results = {}

    for arch_key, arch_info in ARCHITECTURES.items():
        print(f"\n{'#'*70}")
        print(f"Architecture: {arch_info['name']}")
        print(f"{'#'*70}")

        # Reset seeds for fair comparison
        set_random_seeds(config.MASTER_SEED, verbose=False)

        # Train
        model, history, training_time = train_single_architecture(
            arch_key=arch_key,
            arch_info=arch_info,
            X_train=X_train_proc,
            y_train=y_train,
            X_val=X_val_proc,
            y_val=y_val,
            verbose=1
        )

        # Evaluate on all SNR
        print(f"\nEvaluating {arch_info['name']}:")
        snr_accuracy = {}
        for snr in sorted(X_test_dict_proc.keys()):
            y_pred = predict(model, X_test_dict_proc[snr])
            acc = float(accuracy_score(y_test_dict[snr], y_pred))
            snr_accuracy[snr] = acc
            print(f"  SNR {snr:2d} dB: {acc*100:.2f}%")

        # Measure inference time
        print(f"\nMeasuring inference time:")
        inference_time = measure_inference_time(
            model=model,
            X=X_test_dict_proc[config.TRAIN_SNR],
            num_runs=10,
            verbose=True
        )

        # Store results
        best_val_acc = float(max(history['val_accuracy']) * 100)

        arch_results[arch_key] = {
            'model': model,
            'history': history,
            'training_time': training_time,
            'snr_accuracy': snr_accuracy,
            'inference_time': inference_time,
            'best_val_accuracy': best_val_acc
        }

    # =========================================================================
    # STEP 4: Generate Plots
    # =========================================================================

    print("\n" + "=" * 70)
    print("STEP 4: Generating Plots")
    print("=" * 70)

    # Architecture comparison
    print("\n1. Architecture accuracy comparison...")
    plot_architecture_comparison(
        arch_results=arch_results,
        save_path=os.path.join(ARCH_PLOTS_DIR, "architecture_comparison.png"),
        show_plot=show_plots
    )

    # Complexity vs accuracy
    print("2. Complexity vs accuracy...")
    plot_complexity_vs_accuracy(
        arch_results=arch_results,
        all_complexity=all_complexity,
        save_path=os.path.join(ARCH_PLOTS_DIR, "complexity_vs_accuracy.png"),
        show_plot=show_plots
    )

    # =========================================================================
    # STEP 5: Save Results
    # =========================================================================

    print("\n" + "=" * 70)
    print("STEP 5: Saving Results")
    print("=" * 70)

    results_df = save_architecture_results(
        arch_results=arch_results,
        all_complexity=all_complexity,
        verbose=True
    )

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================

    pipeline_time = time.time() - pipeline_start

    print("\n" + "#" * 70)
    print("#" + " " * 10 + "ARCHITECTURE COMPARISON COMPLETE" + " " * 27 + "#")
    print("#" * 70)

    print(f"\nTotal time: {pipeline_time:.1f}s ({pipeline_time/60:.2f} min)")

    print(f"\n--- Architecture Summary ---")
    print(f"{'Architecture':<25}{'Params':<10}{'vs DFT':<10}"
          f"{'Infer(ms)':<12}{'Acc@10dB':<10}")
    print("-" * 67)

    for arch_key in ARCHITECTURES.keys():
        if arch_key in arch_results:
            c = all_complexity[arch_key]
            r = arch_results[arch_key]
            print(f"{ARCHITECTURES[arch_key]['name']:<25}"
                  f"{c['total_params']:<10,}"
                  f"{c['dft_multiplication_ratio']:<10.1f}x"
                  f"{r['inference_time']['avg_per_sample_ms']:<12.4f}"
                  f"{r['snr_accuracy'].get(10, 0)*100:<10.2f}%")

    print("-" * 67)

    print(f"\n--- Output Files ---")
    print(f"Results: {ARCH_RESULTS_DIR}")
    print(f"Models: {ARCH_MODELS_DIR}")
    print(f"Plots: {ARCH_PLOTS_DIR}")

    print("\n" + "#" * 70 + "\n")

    return {
        'arch_results': arch_results,
        'all_complexity': all_complexity,
        'results_df': results_df,
        'pipeline_time': pipeline_time
    }


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    show_plots = True

    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg == "--no-plots":
                show_plots = False
            elif arg == "--help":
                print("\nUsage: python main_architectures.py [options]")
                print("\nOptions:")
                print("  --no-plots    Run without displaying plots")
                print("  --help        Show this help message")
                sys.exit(0)

    results = run_architecture_comparison(show_plots=show_plots)

    if results is not None:
        print("Architecture comparison completed successfully!")
    else:
        print("Architecture comparison failed!")
        sys.exit(1)
