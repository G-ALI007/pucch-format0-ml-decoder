"""
================================================================================
PUCCH Format 0 - Two-Stage ML Decoder with DTX Detection
================================================================================

Novel Contribution:
    Two-stage approach for PUCCH Format 0 decoding with DTX detection:
    
    Stage 1: Binary classifier (UCI vs DTX)
        - Detects whether a PUCCH signal was transmitted or not
        - 2 classes: UCI (0) and DTX (1)
    
    Stage 2: UCI classifier (4 classes)
        - Classifies the UCI content (only if Stage 1 detects UCI)
        - 4 classes: ACK/NACK combinations with SR
        - Uses the pre-trained 4-class model from the base experiment

    This approach separates the detection and classification tasks,
    allowing each stage to focus on a simpler problem.

Comparison Systems:
    System A: 4-class base model (no DTX detection)
    System B: 5-class single model (from main_dtx.py)
    System C: Two-stage model (this script) <- Novel

Usage:
    python main_twostage.py
    python main_twostage.py --no-plots

================================================================================
"""

from main_dtx import (
    load_uci_dataset,
    load_dtx_dataset,
    load_merged_dataset,
    load_all_merged_datasets,
    compute_dtx_metrics
)
from visualization import plot_training_history
from model import (
    set_random_seeds,
    create_model,
    load_saved_model,
    get_training_summary,
    predict,
    print_tensorflow_info
)
from data_loader import analyze_dataset
from config_dtx import config_dtx
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from tf_keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    CSVLogger
)
from tf_keras.utils import to_categorical
import tf_keras
import os
import sys
import time
import datetime
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional

# Fix for TensorFlow >= 2.16
os.environ['TF_USE_LEGACY_KERAS'] = '1'

# tf_keras imports

# Scikit-learn

# Visualization

# Import configurations

# Import from existing modules

# Import from DTX module


# =============================================================================
# SECTION 1: CONFIGURATION FOR TWO-STAGE
# =============================================================================

# Output directories (separate from other experiments)
TWOSTAGE_RESULTS_DIR = "./results_twostage/"
TWOSTAGE_MODELS_DIR = "./models_twostage/"
TWOSTAGE_PLOTS_DIR = "./plots_twostage/"
TWOSTAGE_LOGS_DIR = "./logs_twostage/"

# Model filenames
STAGE1_MODEL_FILENAME = "stage1_dtx_detector.h5"
STAGE2_MODEL_FILENAME = "stage2_uci_classifier.h5"

# Stage 1 configuration
STAGE1_NUM_CLASSES = 2   # UCI=0, DTX=1
STAGE1_OUTPUT_SIZE = 2

# Stage 2 configuration (same as base 4-class model)
STAGE2_NUM_CLASSES = 4
STAGE2_OUTPUT_SIZE = 4


def create_twostage_directories():
    """Create output directories for two-stage experiment."""
    for directory in [TWOSTAGE_RESULTS_DIR, TWOSTAGE_MODELS_DIR,
                      TWOSTAGE_PLOTS_DIR, TWOSTAGE_LOGS_DIR]:
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created: {directory}")


# =============================================================================
# SECTION 2: DATA PREPARATION FOR TWO STAGES
# =============================================================================

def prepare_stage1_data(
    X: np.ndarray,
    y: np.ndarray,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare data for Stage 1: Binary classification (UCI vs DTX).

    Converts multi-class labels to binary:
        Classes 0, 1, 2, 3 (UCI) -> 0
        Class 4 (DTX)            -> 1

    Parameters:
    -----------
    X : np.ndarray
        Features, shape (num_samples, 24)
    y : np.ndarray
        Original labels (0-4), shape (num_samples,)
    verbose : bool, default=True
        Print information

    Returns:
    --------
    X : np.ndarray
        Same features (unchanged)
    y_binary : np.ndarray
        Binary labels: 0=UCI, 1=DTX
    """

    dtx_class = config_dtx.DTX_CLASS  # = 4

    # Convert: classes 0-3 -> 0 (UCI), class 4 -> 1 (DTX)
    y_binary = (y == dtx_class).astype(np.int32)

    if verbose:
        uci_count = int(np.sum(y_binary == 0))
        dtx_count = int(np.sum(y_binary == 1))
        print(f"Stage 1 labels: UCI={uci_count:,}, DTX={dtx_count:,}")

    return X, y_binary


def prepare_stage2_data(
    X: np.ndarray,
    y: np.ndarray,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare data for Stage 2: UCI classification (4 classes).

    Filters out DTX samples, keeping only UCI samples (classes 0-3).

    Parameters:
    -----------
    X : np.ndarray
        Features, shape (num_samples, 24)
    y : np.ndarray
        Original labels (0-4), shape (num_samples,)
    verbose : bool, default=True
        Print information

    Returns:
    --------
    X_uci : np.ndarray
        UCI-only features
    y_uci : np.ndarray
        UCI-only labels (0-3)
    """

    dtx_class = config_dtx.DTX_CLASS  # = 4

    # Keep only UCI samples
    uci_mask = (y != dtx_class)
    X_uci = X[uci_mask]
    y_uci = y[uci_mask]

    if verbose:
        print(f"Stage 2 data: {len(y_uci):,} UCI samples")
        unique, counts = np.unique(y_uci, return_counts=True)
        for u, c in zip(unique, counts):
            print(f"  Class {u}: {c:,}")

    return X_uci, y_uci
# -------------------
# new section
# =============================================================================
# SECTION NEW: MULTI-SNR TRAINING DATA PREPARATION
# =============================================================================


def prepare_multisn_training_data(
    datasets: Dict[int, Tuple[np.ndarray, np.ndarray]],
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """
    Prepare training data by combining ALL SNR values.

    Instead of training on a single SNR, we combine data from all SNR values
    to create a more robust classifier that works across all conditions.

    For each SNR value, we take an equal number of samples to ensure
    balanced representation of different channel conditions.

    Parameters:
    -----------
    datasets : dict
        Dictionary mapping SNR to (X, y) tuples
        Each dataset has 250,000 samples (50,000 per class × 5 classes)
    verbose : bool, default=True
        Print progress

    Returns:
    --------
    X_train : np.ndarray
        Combined training features from all SNR values
    y_train : np.ndarray
        Combined training labels
    X_val : np.ndarray
        Combined validation features
    y_val : np.ndarray
        Combined validation labels
    X_test_dict : dict
        Test features per SNR (same as input, for independent testing)
    y_test_dict : dict
        Test labels per SNR
    """

    if verbose:
        print("\n" + "=" * 70)
        print("PREPARING MULTI-SNR TRAINING DATA")
        print("=" * 70)

    all_X_train = []
    all_y_train = []
    all_X_val = []
    all_y_val = []

    X_test_dict = {}
    y_test_dict = {}

    for snr in sorted(datasets.keys()):
        X_full, y_full = datasets[snr]

        # Split each SNR dataset into train/val
        X_train_snr, X_val_snr, y_train_snr, y_val_snr = train_test_split(
            X_full, y_full,
            train_size=config_dtx.TRAIN_RATIO,
            random_state=config_dtx.SKLEARN_SEED,
            stratify=y_full
        )

        all_X_train.append(X_train_snr)
        all_y_train.append(y_train_snr)
        all_X_val.append(X_val_snr)
        all_y_val.append(y_val_snr)

        # Use full dataset for testing (independent evaluation per SNR)
        X_test_dict[snr] = X_full
        y_test_dict[snr] = y_full

        if verbose:
            print(
                f"  SNR {snr:2d} dB: train={len(X_train_snr):,}, val={len(X_val_snr):,}")

    # Combine all SNR data
    X_train = np.vstack(all_X_train)
    y_train = np.concatenate(all_y_train)
    X_val = np.vstack(all_X_val)
    y_val = np.concatenate(all_y_val)

    # Shuffle combined data
    train_shuffle = np.random.permutation(len(y_train))
    X_train = X_train[train_shuffle]
    y_train = y_train[train_shuffle]

    val_shuffle = np.random.permutation(len(y_val))
    X_val = X_val[val_shuffle]
    y_val = y_val[val_shuffle]

    if verbose:
        print(f"\nCombined training: {len(X_train):,} samples")
        print(f"Combined validation: {len(X_val):,} samples")

        print(f"\nTraining class distribution:")
        unique, counts = np.unique(y_train, return_counts=True)
        for u, c in zip(unique, counts):
            label = config_dtx.CLASS_LABELS.get(int(u), f"Class {u}")
            print(f"  Class {u}: {c:,} samples - {label}")

        print(f"\nTest datasets (per-SNR, independent):")
        for snr in sorted(X_test_dict.keys()):
            print(f"  SNR {snr:2d} dB: {len(y_test_dict[snr]):,} samples")

    if verbose:
        print("=" * 70 + "\n")

    return X_train, y_train, X_val, y_val, X_test_dict, y_test_dict

# =============================================================================
# SECTION 3: STAGE 1 - DTX DETECTOR (BINARY)
# =============================================================================


def create_stage1_model(verbose: bool = True):
    """
    Create Stage 1 model: Binary classifier (UCI vs DTX).

    Architecture: Same as base model but with 2 output neurons.

    Returns:
    --------
    model : keras Sequential model
    """

    if verbose:
        print("\n" + "=" * 70)
        print("CREATING STAGE 1 MODEL (DTX Detector)")
        print("=" * 70)

    model = create_model(
        input_size=config_dtx.INPUT_SIZE,
        hidden_layers=config_dtx.HIDDEN_LAYERS,
        output_size=STAGE1_OUTPUT_SIZE,  # 2 classes
        hidden_activation=config_dtx.HIDDEN_ACTIVATION,
        output_activation=config_dtx.OUTPUT_ACTIVATION,
        dropout_rate=config_dtx.DROPOUT_RATE,
        use_dropout=config_dtx.USE_DROPOUT,
        kernel_initializer=config_dtx.KERNEL_INITIALIZER,
        learning_rate=config_dtx.LEARNING_RATE,
        momentum=config_dtx.MOMENTUM,
        use_nesterov=config_dtx.USE_NESTEROV,
        print_summary=verbose
    )

    return model


def train_stage1(
    model,
    X_train: np.ndarray,
    y_train_binary: np.ndarray,
    X_val: np.ndarray,
    y_val_binary: np.ndarray,
    verbose: int = 1
) -> Tuple[Dict, float]:
    """
    Train Stage 1: Binary DTX detector.

    Parameters:
    -----------
    model : keras model
        Stage 1 model (2 output neurons)
    X_train : np.ndarray
        Training features
    y_train_binary : np.ndarray
        Binary training labels (0=UCI, 1=DTX)
    X_val : np.ndarray
        Validation features
    y_val_binary : np.ndarray
        Binary validation labels
    verbose : int, default=1
        Verbosity level

    Returns:
    --------
    history : dict
        Training history
    training_time : float
        Training time in seconds
    """

    print("\n" + "=" * 70)
    print("TRAINING STAGE 1 (DTX Detector - Binary)")
    print("=" * 70)

    # Convert to one-hot (2 classes)
    y_train_onehot = to_categorical(
        y_train_binary, num_classes=STAGE1_NUM_CLASSES)
    y_val_onehot = to_categorical(y_val_binary, num_classes=STAGE1_NUM_CLASSES)

    model_filepath = os.path.join(TWOSTAGE_MODELS_DIR, STAGE1_MODEL_FILENAME)
    history_filepath = os.path.join(
        TWOSTAGE_LOGS_DIR, "stage1_training_history.csv")

    # Print info
    print(f"\nTraining samples: {len(X_train):,}")
    print(f"Validation samples: {len(X_val):,}")
    print(f"Classes: {STAGE1_NUM_CLASSES} (UCI=0, DTX=1)")

    # Class distribution
    for name, y_set in [("Train", y_train_binary), ("Val", y_val_binary)]:
        unique, counts = np.unique(y_set, return_counts=True)
        dist = " ".join(
            [f"{'UCI' if u == 0 else 'DTX'}:{c:,}" for u, c in zip(unique, counts)])
        print(f"  {name}: {dist}")

    # Ensure directories exist
    for path in [model_filepath, history_filepath]:
        d = os.path.dirname(path)
        if d and not os.path.exists(d):
            os.makedirs(d)

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            filepath=model_filepath,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            mode='min',
            patience=config_dtx.EARLY_STOPPING_PATIENCE,
            min_delta=config_dtx.EARLY_STOPPING_MIN_DELTA,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            mode='min',
            factor=config_dtx.REDUCE_LR_FACTOR,
            patience=config_dtx.REDUCE_LR_PATIENCE,
            min_lr=config_dtx.REDUCE_LR_MIN_LR,
            verbose=1
        ),
        CSVLogger(
            filename=history_filepath,
            separator=',',
            append=False
        )
    ]

    print(f"\nModel path: {model_filepath}")
    print(f"\nStarting training...")
    print("-" * 70)

    start_time = time.time()

    history_obj = model.fit(
        x=X_train,
        y=y_train_onehot,
        validation_data=(X_val, y_val_onehot),
        epochs=config_dtx.NUM_EPOCHS,
        batch_size=config_dtx.BATCH_SIZE,
        callbacks=callbacks,
        verbose=verbose
    )

    training_time = time.time() - start_time
    history = history_obj.history

    print("-" * 70)
    print(f"\nStage 1 training complete!")
    print(f"Time: {training_time:.1f}s ({training_time/60:.2f} min)")
    print(f"Final train acc: {history['accuracy'][-1]*100:.2f}%")
    print(f"Final val acc: {history['val_accuracy'][-1]*100:.2f}%")

    best_epoch = int(np.argmax(history['val_accuracy']) + 1)
    best_acc = float(max(history['val_accuracy']) * 100)
    print(f"Best val acc: {best_acc:.2f}% at epoch {best_epoch}")
    print("=" * 70 + "\n")

    return history, training_time


# =============================================================================
# SECTION 4: TWO-STAGE PREDICTION
# =============================================================================

def predict_twostage(
    stage1_model,
    stage2_model,
    X: np.ndarray,
    verbose: bool = False
) -> np.ndarray:
    """
    Make predictions using the two-stage approach.

    Stage 1: Classify as UCI (0) or DTX (1)
    Stage 2: If UCI, classify into classes 0-3
    Final output: 0, 1, 2, 3 (UCI) or 4 (DTX)

    Parameters:
    -----------
    stage1_model : keras model
        Binary DTX detector (2 outputs)
    stage2_model : keras model
        UCI classifier (4 outputs)
    X : np.ndarray
        Input features, shape (num_samples, 24)
    verbose : bool, default=False
        Print progress

    Returns:
    --------
    y_pred : np.ndarray
        Predicted labels (0-4), shape (num_samples,)
    """

    num_samples = len(X)

    if verbose:
        print(f"Two-stage prediction on {num_samples:,} samples...")

    # Stage 1: DTX detection (binary)
    stage1_proba = stage1_model.predict(X, batch_size=256, verbose=0)
    stage1_pred = np.argmax(stage1_proba, axis=1)  # 0=UCI, 1=DTX

    # Initialize final predictions
    y_pred = np.full(num_samples, config_dtx.DTX_CLASS,
                     dtype=np.int32)  # Default: DTX (4)

    # Find samples classified as UCI by Stage 1
    uci_mask = (stage1_pred == 0)
    uci_indices = np.where(uci_mask)[0]

    if verbose:
        print(
            f"  Stage 1: UCI={np.sum(uci_mask):,}, DTX={np.sum(~uci_mask):,}")

    # Stage 2: Classify UCI samples
    if len(uci_indices) > 0:
        X_uci = X[uci_indices]
        stage2_proba = stage2_model.predict(X_uci, batch_size=256, verbose=0)
        stage2_pred = np.argmax(stage2_proba, axis=1)  # 0, 1, 2, or 3

        # Assign UCI predictions
        y_pred[uci_indices] = stage2_pred

        if verbose:
            unique, counts = np.unique(stage2_pred, return_counts=True)
            dist = " ".join([f"C{u}:{c:,}" for u, c in zip(unique, counts)])
            print(f"  Stage 2: {dist}")

    return y_pred


# =============================================================================
# SECTION 5: TWO-STAGE EVALUATION
# =============================================================================

def evaluate_twostage_all_snr(
    stage1_model,
    stage2_model,
    X_test_dict: Dict[int, np.ndarray],
    y_test_dict: Dict[int, np.ndarray],
    verbose: bool = True
) -> Dict[int, Dict]:
    """
    Evaluate two-stage model on all SNR values.

    Parameters:
    -----------
    stage1_model : keras model
        Binary DTX detector
    stage2_model : keras model
        UCI classifier
    X_test_dict : dict
        Test features per SNR
    y_test_dict : dict
        Test labels per SNR (0-4)
    verbose : bool, default=True
        Print results

    Returns:
    --------
    all_results : dict mapping SNR -> results dict
    """

    if verbose:
        print("\n" + "=" * 70)
        print("EVALUATING TWO-STAGE MODEL")
        print("=" * 70)

    all_results = {}

    for snr in sorted(X_test_dict.keys()):
        X_test = X_test_dict[snr]
        y_test = y_test_dict[snr]

        if verbose:
            print(f"\n{'='*30} SNR = {snr} dB {'='*30}")

        # Two-stage prediction
        y_pred = predict_twostage(
            stage1_model, stage2_model, X_test, verbose=verbose)

        # Compute standard metrics
        accuracy = float(accuracy_score(y_test, y_pred))
        conf_matrix = confusion_matrix(
            y_test, y_pred, labels=list(range(config_dtx.NUM_CLASSES)))

        # Compute DTX-specific metrics
        dtx_metrics = compute_dtx_metrics(y_test, y_pred, verbose=verbose)

        # Store results
        results = {
            'snr_db': snr,
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix,
            'per_class_accuracy': dtx_metrics['per_class_accuracy'],
            'y_true': y_test,
            'y_pred': y_pred,
            'num_samples': len(y_test),
            'num_correct': int(np.sum(y_test == y_pred)),
            'num_errors': int(np.sum(y_test != y_pred)),
            'dtx_metrics': dtx_metrics
        }

        all_results[snr] = results

    # Summary table
    if verbose:
        print("\n\n" + "=" * 95)
        print(f"{'SNR':<6}{'Overall':<10}{'UCI Acc':<10}{'DTX Det':<10}"
              f"{'FalseAlarm':<12}{'MissedDet':<12}{'3GPP FA':<10}{'3GPP MD':<10}")
        print("-" * 95)

        for snr in sorted(all_results.keys()):
            r = all_results[snr]
            dm = r['dtx_metrics']
            print(f"{snr:<6}"
                  f"{r['accuracy']*100:<10.2f}"
                  f"{dm['uci_accuracy']*100:<10.2f}"
                  f"{dm['dtx_detection_rate']*100:<10.2f}"
                  f"{dm['false_alarm_rate']*100:<12.4f}"
                  f"{dm['missed_detection_rate']*100:<12.4f}"
                  f"{'PASS' if dm['false_alarm_passes_3gpp'] else 'FAIL':<10}"
                  f"{'PASS' if dm['missed_detection_passes_3gpp'] else 'FAIL':<10}")

        print("-" * 95)
        print("=" * 95 + "\n")

    return all_results


# =============================================================================
# SECTION 6: THREE-SYSTEM COMPARISON
# =============================================================================

def compare_three_systems(
    results_4class: Dict,
    results_5class: Dict,
    results_twostage: Dict[int, Dict],
    verbose: bool = True
) -> Dict:
    """
    Compare all three systems:
        System A: 4-class base (no DTX)
        System B: 5-class single model
        System C: Two-stage model

    Parameters:
    -----------
    results_4class : dict
        SNR -> accuracy (%) for 4-class model
    results_5class : dict
        SNR -> accuracy (%) for 5-class model
    results_twostage : dict
        SNR -> results dict for two-stage model
    verbose : bool
        Print comparison

    Returns:
    --------
    comparison : dict
    """

    if verbose:
        print("\n" + "=" * 70)
        print("THREE-SYSTEM COMPARISON")
        print("=" * 70)

    comparison = {
        'snr': [],
        'system_a_4class': [],
        'system_b_5class': [],
        'system_c_twostage_overall': [],
        'system_c_twostage_uci': [],
        'system_c_dtx_detection': [],
        'system_c_false_alarm': [],
        'system_c_missed_detection': []
    }

    if verbose:
        print(f"\n{'SNR':<6}{'A:4-Class':<12}{'B:5-Class':<12}{'C:2Stage':<12}"
              f"{'C:UCI Acc':<12}{'C:DTX Det':<12}{'C:FA Rate':<12}")
        print("-" * 78)

    for snr in sorted(results_twostage.keys()):
        r_ts = results_twostage[snr]
        dm = r_ts['dtx_metrics']

        acc_4class = float(results_4class.get(snr, 0.0))
        acc_5class = float(results_5class.get(snr, 0.0))

        comparison['snr'].append(snr)
        comparison['system_a_4class'].append(acc_4class)
        comparison['system_b_5class'].append(acc_5class)
        comparison['system_c_twostage_overall'].append(
            float(r_ts['accuracy'] * 100))
        comparison['system_c_twostage_uci'].append(
            float(dm['uci_accuracy'] * 100))
        comparison['system_c_dtx_detection'].append(
            float(dm['dtx_detection_rate'] * 100))
        comparison['system_c_false_alarm'].append(
            float(dm['false_alarm_rate'] * 100))
        comparison['system_c_missed_detection'].append(
            float(dm['missed_detection_rate'] * 100))

        if verbose:
            print(f"{snr:<6}{acc_4class:<12.2f}{acc_5class:<12.2f}"
                  f"{r_ts['accuracy']*100:<12.2f}{dm['uci_accuracy']*100:<12.2f}"
                  f"{dm['dtx_detection_rate']*100:<12.2f}{dm['false_alarm_rate']*100:<12.4f}")

    if verbose:
        print("-" * 78)
        print("=" * 70 + "\n")

    return comparison


# =============================================================================
# SECTION 7: VISUALIZATION
# =============================================================================

def plot_three_system_comparison(
    comparison: Dict,
    save_path: Optional[str] = None,
    show_plot: bool = True
) -> None:
    """
    Plot comparison of all three systems.

    Parameters:
    -----------
    comparison : dict
        Three-system comparison data
    save_path : str, optional
    show_plot : bool
    """

    if save_path is None:
        save_path = os.path.join(
            TWOSTAGE_PLOTS_DIR, "three_system_comparison.png")

    snr_values = comparison['snr']

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: UCI Accuracy comparison
    axes[0].plot(snr_values, comparison['system_a_4class'], 'b-o',
                 linewidth=2.5, markersize=10, label='System A: 4-Class',
                 markerfacecolor='white', markeredgewidth=2)
    axes[0].plot(snr_values, comparison['system_b_5class'], 'r--s',
                 linewidth=2.5, markersize=10, label='System B: 5-Class',
                 markerfacecolor='white', markeredgewidth=2)
    axes[0].plot(snr_values, comparison['system_c_twostage_uci'], 'g-.^',
                 linewidth=2.5, markersize=10, label='System C: Two-Stage (UCI)',
                 markerfacecolor='white', markeredgewidth=2)

    axes[0].axhline(y=99, color='gray', linestyle=':',
                    linewidth=1.5, alpha=0.7)
    axes[0].set_xlabel('SNR (dB)', fontsize=14)
    axes[0].set_ylabel('UCI Accuracy (%)', fontsize=14)
    axes[0].set_title('UCI Classification Accuracy', fontsize=16)
    axes[0].legend(fontsize=10, loc='lower right')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(snr_values)

    all_uci = (comparison['system_a_4class'] +
               comparison['system_b_5class'] +
               comparison['system_c_twostage_uci'])
    axes[0].set_ylim([max(0, min(all_uci) - 5), 102])

    # Right: DTX metrics (Systems B and C)
    axes[1].plot(snr_values, comparison['system_c_false_alarm'], 'r-o',
                 linewidth=2.5, markersize=10, label='Two-Stage: False Alarm',
                 markerfacecolor='white', markeredgewidth=2)
    axes[1].plot(snr_values, comparison['system_c_missed_detection'], 'b-s',
                 linewidth=2.5, markersize=10, label='Two-Stage: Missed Detection',
                 markerfacecolor='white', markeredgewidth=2)
    axes[1].plot(snr_values, comparison['system_c_dtx_detection'], 'g-^',
                 linewidth=2.5, markersize=10, label='Two-Stage: DTX Detection',
                 markerfacecolor='white', markeredgewidth=2)

    axes[1].axhline(y=1.0, color='orange', linestyle=':', linewidth=2,
                    label='3GPP Requirement (1%)', alpha=0.8)

    axes[1].set_xlabel('SNR (dB)', fontsize=14)
    axes[1].set_ylabel('Rate (%)', fontsize=14)
    axes[1].set_title('DTX Detection Metrics (Two-Stage)', fontsize=16)
    axes[1].legend(fontsize=9, loc='center right')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(snr_values)

    plt.tight_layout()

    directory = os.path.dirname(save_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    fig.savefig(save_path, dpi=config_dtx.FIGURE_DPI, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def plot_twostage_confusion_matrices(
    all_results: Dict[int, Dict],
    save_path: Optional[str] = None,
    show_plot: bool = True
) -> None:
    """
    Plot confusion matrices for two-stage model.
    """

    if save_path is None:
        save_path = os.path.join(
            TWOSTAGE_PLOTS_DIR, "confusion_matrices_twostage.png")

    snr_values = sorted(all_results.keys())
    num_snr = len(snr_values)

    fig, axes = plt.subplots(1, num_snr, figsize=(5 * num_snr, 4.5))

    if num_snr == 1:
        axes = [axes]

    class_labels = [f'C{i}' for i in range(config_dtx.NUM_CLASSES)]

    for idx, snr in enumerate(snr_values):
        conf_matrix = all_results[snr]['confusion_matrix']
        accuracy = all_results[snr]['accuracy'] * 100

        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_labels,
            yticklabels=class_labels,
            ax=axes[idx],
            cbar=True,
            annot_kws={"size": 8},
            linewidths=0.5,
            linecolor='gray'
        )

        axes[idx].set_xlabel('Predicted', fontsize=10)
        axes[idx].set_ylabel('Actual', fontsize=10)
        axes[idx].set_title(f'SNR={snr}dB\nAcc={accuracy:.1f}%', fontsize=11)

    plt.tight_layout()

    directory = os.path.dirname(save_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    fig.savefig(save_path, dpi=config_dtx.FIGURE_DPI, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


# =============================================================================
# SECTION 8: SAVE RESULTS
# =============================================================================

def save_twostage_results(
    all_results: Dict[int, Dict],
    comparison: Dict,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Save two-stage experiment results.
    """

    filepath = os.path.join(TWOSTAGE_RESULTS_DIR,
                            "results_summary_twostage.csv")

    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    data = []
    for i, snr in enumerate(comparison['snr']):
        dm = all_results[snr]['dtx_metrics']

        row = {
            'SNR_dB': snr,
            'System_A_4Class_pct': comparison['system_a_4class'][i],
            'System_B_5Class_pct': comparison['system_b_5class'][i],
            'System_C_TwoStage_Overall_pct': comparison['system_c_twostage_overall'][i],
            'System_C_TwoStage_UCI_pct': comparison['system_c_twostage_uci'][i],
            'System_C_DTX_Detection_pct': comparison['system_c_dtx_detection'][i],
            'System_C_False_Alarm_pct': comparison['system_c_false_alarm'][i],
            'System_C_Missed_Detection_pct': comparison['system_c_missed_detection'][i],
            '3GPP_FA_Pass': dm['false_alarm_passes_3gpp'],
            '3GPP_MD_Pass': dm['missed_detection_passes_3gpp']
        }
        data.append(row)

    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False, float_format='%.4f')

    if verbose:
        print(f"\nResults saved to: {filepath}")
        print("\nThree-System Comparison:")
        print("-" * 90)
        display_cols = ['SNR_dB', 'System_A_4Class_pct', 'System_B_5Class_pct',
                        'System_C_TwoStage_UCI_pct', 'System_C_DTX_Detection_pct',
                        'System_C_False_Alarm_pct']
        print(df[display_cols].to_string(index=False))
        print("-" * 90)

    return df


# =============================================================================
# SECTION 9: MAIN PIPELINE
# =============================================================================

def run_twostage_pipeline(show_plots: bool = True):
    """
    Run the complete two-stage pipeline with multi-SNR training.
    """

    pipeline_start = time.time()

    print("\n" + "#" * 70)
    print("#" + " " * 12 + "TWO-STAGE DTX PIPELINE (Multi-SNR)" + " " * 22 + "#")
    print("#" * 70)

    print(
        f"\nTimestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print_tensorflow_info()

    create_twostage_directories()
    set_random_seeds(config_dtx.MASTER_SEED)

    # =========================================================================
    # STEP 1: Load Data
    # =========================================================================

    print("\n" + "=" * 70)
    print("STEP 1: Loading Merged Datasets")
    print("=" * 70)

    try:
        datasets = load_all_merged_datasets(verbose=True)
    except RuntimeError as e:
        print(f"ERROR: {e}")
        return None

    # =========================================================================
    # STEP 2: Prepare Multi-SNR Training Data
    # =========================================================================

    print("\n" + "=" * 70)
    print("STEP 2: Preparing Multi-SNR Training Data")
    print("=" * 70)

    X_train_all, y_train_all, X_val_all, y_val_all, X_test_dict, y_test_dict = \
        prepare_multisn_training_data(datasets, verbose=True)

    # =========================================================================
    # STEP 3: Prepare Stage 1 Data (Binary: UCI vs DTX)
    # =========================================================================

    print("\n" + "=" * 70)
    print("STEP 3: Preparing Stage 1 Data (UCI vs DTX)")
    print("=" * 70)

    print("\nTraining data:")
    X_train_s1, y_train_s1 = prepare_stage1_data(
        X_train_all, y_train_all, verbose=True)
    print("\nValidation data:")
    X_val_s1, y_val_s1 = prepare_stage1_data(
        X_val_all, y_val_all, verbose=True)

    # =========================================================================
    # STEP 4: Train Stage 1 (DTX Detector)
    # =========================================================================

    print("\n" + "=" * 70)
    print("STEP 4: Training Stage 1 (DTX Detector) - Multi-SNR")
    print("=" * 70)

    stage1_model = create_stage1_model(verbose=True)

    stage1_history, stage1_time = train_stage1(
        model=stage1_model,
        X_train=X_train_s1,
        y_train_binary=y_train_s1,
        X_val=X_val_s1,
        y_val_binary=y_val_s1,
        verbose=1
    )

    # Load best Stage 1 model
    stage1_filepath = os.path.join(TWOSTAGE_MODELS_DIR, STAGE1_MODEL_FILENAME)
    try:
        stage1_best = load_saved_model(stage1_filepath)
    except FileNotFoundError:
        stage1_best = stage1_model

    # =========================================================================
    # STEP 5: Prepare and Train Stage 2 (UCI Classifier) - Multi-SNR
    # =========================================================================

    print("\n" + "=" * 70)
    print("STEP 5: Preparing and Training Stage 2 (UCI Classifier)")
    print("=" * 70)

    # Check if pre-trained 4-class model exists
    stage2_pretrained_path = os.path.join(
        "./models/", "pucch_f0_nn_decoder.h5")

    if os.path.exists(stage2_pretrained_path):
        print(f"\nPre-trained 4-class model found: {stage2_pretrained_path}")
        print("However, for fair comparison, we train Stage 2 on multi-SNR data too.")

    # Prepare Stage 2 data (UCI only, all SNR values)
    print("\nPreparing Stage 2 training data (UCI only):")
    X_train_s2, y_train_s2 = prepare_stage2_data(
        X_train_all, y_train_all, verbose=True)
    print("\nPreparing Stage 2 validation data (UCI only):")
    X_val_s2, y_val_s2 = prepare_stage2_data(
        X_val_all, y_val_all, verbose=True)

    # Create Stage 2 model
    print("\nCreating Stage 2 model (4-class UCI classifier):")
    stage2_model = create_model(
        input_size=config_dtx.INPUT_SIZE,
        hidden_layers=config_dtx.HIDDEN_LAYERS,
        output_size=STAGE2_OUTPUT_SIZE,  # 4 classes
        hidden_activation=config_dtx.HIDDEN_ACTIVATION,
        output_activation=config_dtx.OUTPUT_ACTIVATION,
        dropout_rate=config_dtx.DROPOUT_RATE,
        use_dropout=config_dtx.USE_DROPOUT,
        kernel_initializer=config_dtx.KERNEL_INITIALIZER,
        learning_rate=config_dtx.LEARNING_RATE,
        momentum=config_dtx.MOMENTUM,
        use_nesterov=config_dtx.USE_NESTEROV,
        print_summary=True
    )

    # Train Stage 2
    print("\nTraining Stage 2 (4-class UCI classifier):")

    stage2_model_path = os.path.join(
        TWOSTAGE_MODELS_DIR, STAGE2_MODEL_FILENAME)
    stage2_history_path = os.path.join(
        TWOSTAGE_LOGS_DIR, "stage2_training_history.csv")

    # Ensure directories exist
    for path in [stage2_model_path, stage2_history_path]:
        d = os.path.dirname(path)
        if d and not os.path.exists(d):
            os.makedirs(d)

    y_train_s2_oh = to_categorical(y_train_s2, num_classes=STAGE2_NUM_CLASSES)
    y_val_s2_oh = to_categorical(y_val_s2, num_classes=STAGE2_NUM_CLASSES)

    stage2_callbacks = [
        ModelCheckpoint(
            filepath=stage2_model_path,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            mode='min',
            patience=config_dtx.EARLY_STOPPING_PATIENCE,
            min_delta=config_dtx.EARLY_STOPPING_MIN_DELTA,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            mode='min',
            factor=config_dtx.REDUCE_LR_FACTOR,
            patience=config_dtx.REDUCE_LR_PATIENCE,
            min_lr=config_dtx.REDUCE_LR_MIN_LR,
            verbose=1
        ),
        CSVLogger(
            filename=stage2_history_path,
            separator=',',
            append=False
        )
    ]

    print(f"\nStarting Stage 2 training...")
    print("-" * 70)

    stage2_start = time.time()

    stage2_history_obj = stage2_model.fit(
        x=X_train_s2,
        y=y_train_s2_oh,
        validation_data=(X_val_s2, y_val_s2_oh),
        epochs=config_dtx.NUM_EPOCHS,
        batch_size=config_dtx.BATCH_SIZE,
        callbacks=stage2_callbacks,
        verbose=1
    )

    stage2_time = time.time() - stage2_start
    stage2_history = stage2_history_obj.history

    print("-" * 70)
    print(f"\nStage 2 training complete!")
    print(f"Time: {stage2_time:.1f}s ({stage2_time/60:.2f} min)")
    print(f"Best val acc: {max(stage2_history['val_accuracy'])*100:.2f}%")

    # Load best Stage 2 model
    try:
        stage2_best = load_saved_model(stage2_model_path)
    except FileNotFoundError:
        stage2_best = stage2_model

    # =========================================================================
    # STEP 6: Evaluate Two-Stage System
    # =========================================================================

    print("\n" + "=" * 70)
    print("STEP 6: Evaluating Two-Stage System")
    print("=" * 70)

    all_results = evaluate_twostage_all_snr(
        stage1_model=stage1_best,
        stage2_model=stage2_best,
        X_test_dict=X_test_dict,
        y_test_dict=y_test_dict,
        verbose=True
    )

    # =========================================================================
    # STEP 7: Compare Three Systems
    # =========================================================================

    print("\n" + "=" * 70)
    print("STEP 7: Comparing Three Systems")
    print("=" * 70)

    # Load previous results
    results_4class = {}
    base_path = os.path.join("./results/", "results_summary.csv")
    if os.path.exists(base_path):
        df_4class = pd.read_csv(base_path)
        for _, row in df_4class.iterrows():
            results_4class[int(row['SNR_dB'])] = float(row['NN_Accuracy_pct'])
        print(f"Loaded 4-class results from: {base_path}")
    else:
        print("WARNING: 4-class results not found")
        for snr in config_dtx.SNR_VALUES:
            results_4class[snr] = 0.0

    results_5class = {}
    dtx_path = os.path.join("./results_dtx/", "results_summary_dtx.csv")
    if os.path.exists(dtx_path):
        df_5class = pd.read_csv(dtx_path)
        for _, row in df_5class.iterrows():
            results_5class[int(row['SNR_dB'])] = float(
                row['DTX_5Class_Overall_Accuracy_pct'])
        print(f"Loaded 5-class results from: {dtx_path}")
    else:
        print("WARNING: 5-class results not found")
        for snr in config_dtx.SNR_VALUES:
            results_5class[snr] = 0.0

    comparison = compare_three_systems(
        results_4class=results_4class,
        results_5class=results_5class,
        results_twostage=all_results,
        verbose=True
    )

    # =========================================================================
    # STEP 8: Generate Plots
    # =========================================================================

    print("\n" + "=" * 70)
    print("STEP 8: Generating Plots")
    print("=" * 70)

    # Stage 1 training history
    print("\n1. Stage 1 training history...")
    plot_training_history(
        history=stage1_history,
        save_path=os.path.join(
            TWOSTAGE_PLOTS_DIR, "stage1_training_history.png"),
        show_plot=show_plots
    )

    # Stage 2 training history
    print("2. Stage 2 training history...")
    plot_training_history(
        history=stage2_history,
        save_path=os.path.join(
            TWOSTAGE_PLOTS_DIR, "stage2_training_history.png"),
        show_plot=show_plots
    )

    # Three-system comparison
    print("3. Three-system comparison...")
    plot_three_system_comparison(
        comparison=comparison,
        save_path=os.path.join(
            TWOSTAGE_PLOTS_DIR, "three_system_comparison.png"),
        show_plot=show_plots
    )

    # Confusion matrices
    print("4. Confusion matrices...")
    plot_twostage_confusion_matrices(
        all_results=all_results,
        save_path=os.path.join(
            TWOSTAGE_PLOTS_DIR, "confusion_matrices_twostage.png"),
        show_plot=show_plots
    )

    print(f"\nAll plots saved to: {TWOSTAGE_PLOTS_DIR}")

    # =========================================================================
    # STEP 9: Save Results
    # =========================================================================

    print("\n" + "=" * 70)
    print("STEP 9: Saving Results")
    print("=" * 70)

    results_df = save_twostage_results(
        all_results=all_results,
        comparison=comparison,
        verbose=True
    )

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================

    pipeline_time = time.time() - pipeline_start
    stage1_summary = get_training_summary(stage1_history)
    stage2_summary = get_training_summary(stage2_history)

    print("\n" + "#" * 70)
    print("#" + " " * 12 + "TWO-STAGE PIPELINE COMPLETE" + " " * 29 + "#")
    print("#" * 70)

    print(f"\nTotal time: {pipeline_time:.1f}s ({pipeline_time/60:.2f} min)")
    print(
        f"Stage 1 training: {stage1_time:.1f}s (best acc: {stage1_summary['best_val_acc']:.2f}%)")
    print(
        f"Stage 2 training: {stage2_time:.1f}s (best acc: {stage2_summary['best_val_acc']:.2f}%)")

    print(f"\n--- Two-Stage Results (Multi-SNR Training) ---")
    print(f"{'SNR':<6}{'Overall':<10}{'UCI Acc':<10}{'DTX Det':<10}"
          f"{'FA Rate':<12}{'MD Rate':<12}")
    print("-" * 60)
    for snr in sorted(all_results.keys()):
        dm = all_results[snr]['dtx_metrics']
        print(f"{snr:<6}"
              f"{all_results[snr]['accuracy']*100:<10.2f}"
              f"{dm['uci_accuracy']*100:<10.2f}"
              f"{dm['dtx_detection_rate']*100:<10.2f}"
              f"{dm['false_alarm_rate']*100:<12.4f}"
              f"{dm['missed_detection_rate']*100:<12.4f}")
    print("-" * 60)

    print(f"\n--- Output Files ---")
    print(
        f"Stage 1 model: {os.path.join(TWOSTAGE_MODELS_DIR, STAGE1_MODEL_FILENAME)}")
    print(
        f"Stage 2 model: {os.path.join(TWOSTAGE_MODELS_DIR, STAGE2_MODEL_FILENAME)}")
    print(f"Results: {TWOSTAGE_RESULTS_DIR}")
    print(f"Plots: {TWOSTAGE_PLOTS_DIR}")

    print("\n" + "#" * 70 + "\n")

    return {
        'stage1_model': stage1_best,
        'stage2_model': stage2_best,
        'stage1_history': stage1_history,
        'stage2_history': stage2_history,
        'all_results': all_results,
        'comparison': comparison,
        'results_df': results_df,
        'pipeline_time': pipeline_time
    }
# =============================================================================
# SECTION 10: ENTRY POINT
# =============================================================================


if __name__ == "__main__":
    show_plots = True

    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg == "--no-plots":
                show_plots = False
            elif arg == "--help":
                print("\nUsage: python main_twostage.py [options]")
                print("\nOptions:")
                print("  --no-plots    Run without displaying plots")
                print("  --help        Show this help message")
                print("\nTwo-Stage PUCCH Format 0 Decoder:")
                print("  Stage 1: Binary classifier (UCI vs DTX)")
                print("  Stage 2: 4-class UCI classifier (pre-trained)")
                sys.exit(0)

    results = run_twostage_pipeline(show_plots=show_plots)

    if results is not None:
        print("Two-stage pipeline completed successfully!")
    else:
        print("Two-stage pipeline failed!")
        sys.exit(1)
