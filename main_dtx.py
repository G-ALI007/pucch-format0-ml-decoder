"""
================================================================================
PUCCH Format 0 - ML Decoder with DTX Detection
Main Execution Script (5 Classes)
================================================================================

Novel Contribution:
    This script extends the base 4-class PUCCH Format 0 ML decoder to include
    DTX (Discontinuous Transmission) detection as a 5th class.
    
    The original paper stated this as future work:
    "False detections can easily be incorporated into our framework 
    by adding an additional class label whose inputs would be instances of AWGN"

Classes:
    0: ACK=0, SR=0 (NACK, no SR)
    1: ACK=0, SR=1 (NACK, +SR)
    2: ACK=1, SR=0 (ACK, no SR)
    3: ACK=1, SR=1 (ACK, +SR)
    4: DTX (No Transmission)          <- NEW

Usage:
    python main_dtx.py
    python main_dtx.py --no-plots

================================================================================
"""

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
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)
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

# tf_keras imports for training

# Scikit-learn

# Visualization

# Import DTX configuration

# Import from existing modules (functions that work with any number of classes)


# =============================================================================
# SECTION 1: DATA LOADING AND MERGING
# =============================================================================

def load_uci_dataset(snr_db: int, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load UCI dataset for a specific SNR.
    Labels are 0, 1, 2, 3.

    Parameters:
    -----------
    snr_db : int
        SNR value in dB
    verbose : bool, default=True
        Print progress

    Returns:
    --------
    X : np.ndarray, shape (num_samples, 24), dtype float32
    y : np.ndarray, shape (num_samples,), dtype int32

    Raises:
    -------
    FileNotFoundError
        If UCI file does not exist
    """

    filepath = config_dtx.get_uci_filepath(snr_db)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"UCI file not found: {filepath}")

    if verbose:
        print(f"Loading UCI data (SNR={snr_db} dB)...", end=" ", flush=True)

    start_time = time.time()
    df = pd.read_csv(filepath)

    X = df[config_dtx.FEATURE_COLUMNS].values.astype(np.float32)
    y = df[config_dtx.LABEL_COLUMN].values.astype(np.int32)

    elapsed = time.time() - start_time

    if verbose:
        print(
            f"Done! Shape: {X.shape}, Labels: {sorted(np.unique(y))}, Time: {elapsed:.2f}s")

    return X, y


def load_dtx_dataset(snr_db: int, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load DTX dataset for a specific SNR.
    Labels are already 4 in the CSV file.

    Parameters:
    -----------
    snr_db : int
        SNR value in dB
    verbose : bool, default=True
        Print progress

    Returns:
    --------
    X : np.ndarray, shape (num_samples, 24), dtype float32
    y : np.ndarray, shape (num_samples,), dtype int32

    Raises:
    -------
    FileNotFoundError
        If DTX file does not exist
    """

    filepath = config_dtx.get_dtx_filepath(snr_db)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"DTX file not found: {filepath}")

    if verbose:
        print(f"Loading DTX data (SNR={snr_db} dB)...", end=" ", flush=True)

    start_time = time.time()
    df = pd.read_csv(filepath)

    X = df[config_dtx.FEATURE_COLUMNS].values.astype(np.float32)
    y = df[config_dtx.LABEL_COLUMN].values.astype(np.int32)

    elapsed = time.time() - start_time

    if verbose:
        print(
            f"Done! Shape: {X.shape}, Labels: {sorted(np.unique(y))}, Time: {elapsed:.2f}s")

    return X, y


def load_merged_dataset(
    snr_db: int,
    balance_classes: bool = True,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and merge UCI + DTX datasets for a specific SNR.

    Balances classes so each class has equal number of samples.
    UCI has 50,000 samples per class (4 classes = 200,000 total).
    DTX has 50,000 samples (1 class).
    After merging: 50,000 x 5 = 250,000 samples (balanced).

    Parameters:
    -----------
    snr_db : int
        SNR value in dB
    balance_classes : bool, default=True
        If True, ensure all classes have equal samples
    verbose : bool, default=True
        Print progress

    Returns:
    --------
    X : np.ndarray, shape (num_samples, 24), dtype float32
    y : np.ndarray, shape (num_samples,), dtype int32
    """

    # Load UCI data (classes 0-3)
    X_uci, y_uci = load_uci_dataset(snr_db, verbose)

    # Load DTX data (class 4)
    X_dtx, y_dtx = load_dtx_dataset(snr_db, verbose)

    # Balance classes if requested
    if balance_classes:
        # Find minimum samples per class
        uci_unique, uci_counts = np.unique(y_uci, return_counts=True)
        min_per_class = min(min(uci_counts), len(y_dtx))

        if verbose:
            print(f"Balancing classes to {min_per_class:,} samples each...")

        # Sample from UCI classes
        X_uci_balanced = []
        y_uci_balanced = []

        np.random.seed(config_dtx.NUMPY_SEED)

        for c in range(config_dtx.NUM_UCI_CLASSES):
            class_mask = (y_uci == c)
            class_indices = np.where(class_mask)[0]

            if len(class_indices) > min_per_class:
                selected = np.random.choice(
                    class_indices, min_per_class, replace=False)
            else:
                selected = class_indices

            X_uci_balanced.append(X_uci[selected])
            y_uci_balanced.append(y_uci[selected])

        X_uci = np.vstack(X_uci_balanced)
        y_uci = np.concatenate(y_uci_balanced)

        # Sample from DTX class
        if len(y_dtx) > min_per_class:
            dtx_indices = np.random.choice(
                len(y_dtx), min_per_class, replace=False)
            X_dtx = X_dtx[dtx_indices]
            y_dtx = y_dtx[dtx_indices]

    # Merge UCI and DTX
    X_merged = np.vstack([X_uci, X_dtx])
    y_merged = np.concatenate([y_uci, y_dtx])

    # Shuffle
    shuffle_idx = np.random.permutation(len(y_merged))
    X_merged = X_merged[shuffle_idx]
    y_merged = y_merged[shuffle_idx]

    if verbose:
        print(f"Merged dataset: {X_merged.shape[0]:,} samples")
        unique, counts = np.unique(y_merged, return_counts=True)
        for u, c in zip(unique, counts):
            label = config_dtx.CLASS_LABELS.get(int(u), f"Class {u}")
            print(f"  Class {u}: {c:,} samples - {label}")

    return X_merged, y_merged


def load_all_merged_datasets(verbose: bool = True) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """
    Load merged UCI + DTX datasets for all SNR values.

    Returns:
    --------
    datasets : dict mapping SNR (int) -> (X, y) tuple

    Raises:
    -------
    RuntimeError
        If no datasets could be loaded
    """

    if verbose:
        print("\n" + "=" * 70)
        print("LOADING MERGED DATASETS (UCI + DTX)")
        print("=" * 70)

    datasets = {}
    errors = []

    for snr in config_dtx.SNR_VALUES:
        try:
            if verbose:
                print(f"\n--- SNR = {snr} dB ---")

            X, y = load_merged_dataset(
                snr, balance_classes=True, verbose=verbose)
            datasets[snr] = (X, y)

        except FileNotFoundError as e:
            if verbose:
                print(f"  ERROR: {e}")
            errors.append((snr, str(e)))

    if verbose:
        print("\n" + "-" * 70)
        print(
            f"Loaded {len(datasets)}/{len(config_dtx.SNR_VALUES)} merged datasets")
        if errors:
            for snr, msg in errors:
                print(f"  Failed: SNR {snr} dB - {msg}")
        print("=" * 70 + "\n")

    if len(datasets) == 0:
        raise RuntimeError("No datasets could be loaded!")

    return datasets


# =============================================================================
# SECTION 2: TRAINING FUNCTION FOR 5 CLASSES
# =============================================================================

def train_dtx_model(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    verbose: int = 1
) -> Tuple[Dict, float]:
    """
    Train the 5-class model.

    This function is separate from model.py's train_model() because
    that function uses config.NUM_CLASSES=4 for one-hot encoding.
    This function uses config_dtx.NUM_CLASSES=5.

    Parameters:
    -----------
    model : keras Sequential model
        Compiled model with 5 output neurons
    X_train : np.ndarray
        Training features, shape (num_train, 24)
    y_train : np.ndarray
        Training labels, shape (num_train,), values 0-4
    X_val : np.ndarray
        Validation features, shape (num_val, 24)
    y_val : np.ndarray
        Validation labels, shape (num_val,), values 0-4
    verbose : int, default=1
        Verbosity level (0=silent, 1=progress bar, 2=one line per epoch)

    Returns:
    --------
    history : dict
        Training history with keys: 'loss', 'accuracy', 'val_loss', 'val_accuracy'
    training_time : float
        Total training time in seconds
    """

    print("\n" + "=" * 70)
    print("TRAINING 5-CLASS NEURAL NETWORK")
    print("=" * 70)

    # Validate inputs
    if X_train.shape[0] == 0:
        raise ValueError("Training data is empty")
    if X_val.shape[0] == 0:
        raise ValueError("Validation data is empty")
    if X_train.shape[1] != X_val.shape[1]:
        raise ValueError(
            f"Feature mismatch: train has {X_train.shape[1]}, val has {X_val.shape[1]}"
        )
    if len(y_train) != X_train.shape[0]:
        raise ValueError(
            f"X_train/y_train size mismatch: {X_train.shape[0]} vs {len(y_train)}"
        )
    if len(y_val) != X_val.shape[0]:
        raise ValueError(
            f"X_val/y_val size mismatch: {X_val.shape[0]} vs {len(y_val)}"
        )

    # Convert labels to one-hot using 5 CLASSES
    num_classes = config_dtx.NUM_CLASSES  # = 5
    y_train_onehot = to_categorical(y_train, num_classes=num_classes)
    y_val_onehot = to_categorical(y_val, num_classes=num_classes)

    epochs = config_dtx.NUM_EPOCHS
    batch_size = config_dtx.BATCH_SIZE
    model_filepath = config_dtx.get_model_filepath()

    # Print configuration
    print(f"\n--- Training Configuration ---")
    print(f"Training samples: {len(X_train):,}")
    print(f"Validation samples: {len(X_val):,}")
    print(f"Input features: {X_train.shape[1]}")
    print(f"Output classes: {num_classes}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Steps per epoch: {len(X_train) // batch_size}")
    print(f"Model save path: {model_filepath}")

    # Ensure directories exist
    model_dir = os.path.dirname(model_filepath)
    if model_dir and not os.path.exists(model_dir):
        os.makedirs(model_dir)

    history_path = os.path.join(
        config_dtx.LOGS_DIR, config_dtx.TRAINING_HISTORY_FILENAME)
    history_dir = os.path.dirname(history_path)
    if history_dir and not os.path.exists(history_dir):
        os.makedirs(history_dir)

    # Create callbacks
    print("\nCallbacks:")
    callbacks = []

    checkpoint = ModelCheckpoint(
        filepath=model_filepath,
        monitor=config_dtx.CHECKPOINT_MONITOR,
        mode=config_dtx.CHECKPOINT_MODE,
        save_best_only=config_dtx.CHECKPOINT_SAVE_BEST_ONLY,
        verbose=1
    )
    callbacks.append(checkpoint)
    print(f"  - ModelCheckpoint: {model_filepath}")

    early_stopping = EarlyStopping(
        monitor=config_dtx.EARLY_STOPPING_MONITOR,
        mode=config_dtx.EARLY_STOPPING_MODE,
        patience=config_dtx.EARLY_STOPPING_PATIENCE,
        min_delta=config_dtx.EARLY_STOPPING_MIN_DELTA,
        restore_best_weights=config_dtx.EARLY_STOPPING_RESTORE_BEST,
        verbose=1
    )
    callbacks.append(early_stopping)
    print(f"  - EarlyStopping: patience={config_dtx.EARLY_STOPPING_PATIENCE}")

    reduce_lr = ReduceLROnPlateau(
        monitor=config_dtx.REDUCE_LR_MONITOR,
        mode='min',
        factor=config_dtx.REDUCE_LR_FACTOR,
        patience=config_dtx.REDUCE_LR_PATIENCE,
        min_lr=config_dtx.REDUCE_LR_MIN_LR,
        verbose=1
    )
    callbacks.append(reduce_lr)
    print(f"  - ReduceLROnPlateau: factor={config_dtx.REDUCE_LR_FACTOR}")

    csv_logger = CSVLogger(
        filename=history_path,
        separator=',',
        append=False
    )
    callbacks.append(csv_logger)
    print(f"  - CSVLogger: {history_path}")

    # Train
    print(f"\n--- Starting Training ---")
    print("-" * 70)

    start_time = time.time()

    history_obj = model.fit(
        x=X_train,
        y=y_train_onehot,
        validation_data=(X_val, y_val_onehot),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=verbose
    )

    training_time = time.time() - start_time
    history = history_obj.history

    # Print results
    print("-" * 70)
    print(f"\n--- Training Complete ---")
    print(
        f"Total time: {training_time:.1f} seconds ({training_time/60:.2f} minutes)")
    print(f"Epochs completed: {len(history['loss'])}")

    print(f"\n--- Final Metrics ---")
    print(f"Training accuracy: {history['accuracy'][-1]*100:.2f}%")
    print(f"Validation accuracy: {history['val_accuracy'][-1]*100:.2f}%")
    print(f"Training loss: {history['loss'][-1]:.4f}")
    print(f"Validation loss: {history['val_loss'][-1]:.4f}")

    best_epoch = int(np.argmax(history['val_accuracy']) + 1)
    best_val_acc = float(max(history['val_accuracy']) * 100)
    print(f"\n--- Best Results ---")
    print(
        f"Best validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch}")

    print("=" * 70 + "\n")

    return history, training_time


# =============================================================================
# SECTION 3: DTX-SPECIFIC METRICS
# =============================================================================

def compute_dtx_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dtx_class: Optional[int] = None,
    verbose: bool = True
) -> Dict:
    """
    Compute DTX-specific metrics.

    Metrics computed:
        1. False Alarm Rate: P(predict UCI | true DTX)
           Probability of detecting UCI when nothing was sent

        2. Missed Detection Rate: P(predict DTX | true UCI)
           Probability of missing a real UCI transmission

        3. DTX Detection Rate: P(predict DTX | true DTX)
           Probability of correctly identifying no transmission

        4. UCI Classification Accuracy: P(correct UCI | true UCI)
           Accuracy only on UCI samples (excluding DTX)

    Parameters:
    -----------
    y_true : np.ndarray
        True labels, shape (num_samples,)
    y_pred : np.ndarray
        Predicted labels, shape (num_samples,)
    dtx_class : int, optional
        Index of DTX class
        Default: config_dtx.DTX_CLASS (4)
    verbose : bool, default=True
        Print results

    Returns:
    --------
    metrics : dict
        Dictionary containing all DTX-specific metrics
    """

    if dtx_class is None:
        dtx_class = config_dtx.DTX_CLASS

    # Create masks
    true_dtx_mask = (y_true == dtx_class)
    true_uci_mask = (y_true != dtx_class)
    pred_dtx_mask = (y_pred == dtx_class)
    pred_uci_mask = (y_pred != dtx_class)

    # Counts
    num_true_dtx = int(np.sum(true_dtx_mask))
    num_true_uci = int(np.sum(true_uci_mask))
    num_total = len(y_true)

    # 1. False Alarm Rate: predict UCI when true DTX
    if num_true_dtx > 0:
        false_alarm_count = int(np.sum(true_dtx_mask & pred_uci_mask))
        false_alarm_rate = float(false_alarm_count / num_true_dtx)
    else:
        false_alarm_count = 0
        false_alarm_rate = 0.0

    # 2. Missed Detection Rate: predict DTX when true UCI
    if num_true_uci > 0:
        missed_detection_count = int(np.sum(true_uci_mask & pred_dtx_mask))
        missed_detection_rate = float(missed_detection_count / num_true_uci)
    else:
        missed_detection_count = 0
        missed_detection_rate = 0.0

    # 3. DTX Detection Rate: predict DTX when true DTX
    if num_true_dtx > 0:
        dtx_correct_count = int(np.sum(true_dtx_mask & pred_dtx_mask))
        dtx_detection_rate = float(dtx_correct_count / num_true_dtx)
    else:
        dtx_correct_count = 0
        dtx_detection_rate = 0.0

    # 4. UCI Classification Accuracy (only on UCI samples)
    if num_true_uci > 0:
        uci_y_true = y_true[true_uci_mask]
        uci_y_pred = y_pred[true_uci_mask]
        uci_correct = int(np.sum(uci_y_true == uci_y_pred))
        uci_accuracy = float(uci_correct / num_true_uci)
    else:
        uci_correct = 0
        uci_accuracy = 0.0

    # 5. Overall accuracy
    overall_accuracy = float(accuracy_score(y_true, y_pred))

    # 6. Per-class accuracy
    per_class_acc = np.zeros(config_dtx.NUM_CLASSES)
    for c in range(config_dtx.NUM_CLASSES):
        class_mask = (y_true == c)
        if np.sum(class_mask) > 0:
            per_class_acc[c] = float(accuracy_score(
                y_true[class_mask], y_pred[class_mask]))
        else:
            per_class_acc[c] = 0.0

    # 3GPP compliance check
    false_alarm_passes = false_alarm_rate < config_dtx.FALSE_ALARM_REQUIREMENT
    missed_detection_passes = missed_detection_rate < config_dtx.MISSED_DETECTION_REQUIREMENT

    # Store metrics
    metrics = {
        'overall_accuracy': overall_accuracy,
        'uci_accuracy': uci_accuracy,
        'dtx_detection_rate': dtx_detection_rate,
        'false_alarm_rate': false_alarm_rate,
        'false_alarm_count': false_alarm_count,
        'missed_detection_rate': missed_detection_rate,
        'missed_detection_count': missed_detection_count,
        'per_class_accuracy': per_class_acc,
        'num_true_dtx': num_true_dtx,
        'num_true_uci': num_true_uci,
        'num_total': num_total,
        'false_alarm_passes_3gpp': false_alarm_passes,
        'missed_detection_passes_3gpp': missed_detection_passes
    }

    if verbose:
        print(f"\n--- DTX-Specific Metrics ---")
        print(f"Overall Accuracy: {overall_accuracy*100:.2f}%")
        print(f"UCI-only Accuracy: {uci_accuracy*100:.2f}%")
        print(f"DTX Detection Rate: {dtx_detection_rate*100:.2f}%")
        print(f"False Alarm Rate: {false_alarm_rate*100:.4f}% "
              f"({'PASS' if false_alarm_passes else 'FAIL'} 3GPP < {config_dtx.FALSE_ALARM_REQUIREMENT*100}%)")
        print(f"Missed Detection Rate: {missed_detection_rate*100:.4f}% "
              f"({'PASS' if missed_detection_passes else 'FAIL'} 3GPP < {config_dtx.MISSED_DETECTION_REQUIREMENT*100}%)")

    return metrics


# =============================================================================
# SECTION 4: EVALUATION FOR ALL SNR
# =============================================================================

def evaluate_dtx_all_snr(
    model,
    X_test_dict: Dict[int, np.ndarray],
    y_test_dict: Dict[int, np.ndarray],
    verbose: bool = True
) -> Dict[int, Dict]:
    """
    Evaluate the 5-class model on all SNR values with DTX metrics.

    Parameters:
    -----------
    model : keras model
        Trained 5-class model
    X_test_dict : dict
        Dictionary mapping SNR (int) to test features (np.ndarray)
    y_test_dict : dict
        Dictionary mapping SNR (int) to test labels (np.ndarray)
    verbose : bool, default=True
        Print detailed results

    Returns:
    --------
    all_results : dict
        Dictionary mapping SNR (int) to results dictionary
    """

    if verbose:
        print("\n" + "=" * 70)
        print("EVALUATING 5-CLASS MODEL (WITH DTX)")
        print("=" * 70)

    all_results = {}

    for snr in sorted(X_test_dict.keys()):
        X_test = X_test_dict[snr]
        y_test = y_test_dict[snr]

        if verbose:
            print(f"\n{'='*30} SNR = {snr} dB {'='*30}")

        # Get predictions
        y_pred = predict(model, X_test, return_probabilities=False)

        # Compute standard metrics
        accuracy = float(accuracy_score(y_test, y_pred))
        precision = float(precision_score(
            y_test, y_pred, average='weighted', zero_division=0))
        recall_val = float(recall_score(
            y_test, y_pred, average='weighted', zero_division=0))
        f1 = float(f1_score(y_test, y_pred, average='weighted', zero_division=0))
        conf_matrix = confusion_matrix(
            y_test, y_pred, labels=list(range(config_dtx.NUM_CLASSES)))

        # Compute DTX-specific metrics
        dtx_metrics = compute_dtx_metrics(y_test, y_pred, verbose=verbose)

        # Store results
        results = {
            'snr_db': snr,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall_val,
            'f1_score': f1,
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

    # Print summary table
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
# SECTION 5: COMPARISON WITH 4-CLASS MODEL
# =============================================================================

def compare_4class_vs_5class(
    results_4class: Dict,
    results_5class: Dict[int, Dict],
    verbose: bool = True
) -> Dict:
    """
    Compare 4-class (base) results with 5-class (DTX) results.

    Parameters:
    -----------
    results_4class : dict
        Dictionary mapping SNR to 4-class accuracy (%)
    results_5class : dict
        Dictionary mapping SNR to 5-class evaluation results
    verbose : bool, default=True
        Print comparison table

    Returns:
    --------
    comparison : dict
        Dictionary containing comparison data for all SNR values
    """

    if verbose:
        print("\n" + "=" * 70)
        print("COMPARISON: 4-CLASS (Base) vs 5-CLASS (DTX)")
        print("=" * 70)

    comparison = {
        'snr': [],
        'base_4class_accuracy': [],
        'dtx_5class_overall_accuracy': [],
        'dtx_5class_uci_accuracy': [],
        'dtx_detection_rate': [],
        'false_alarm_rate': [],
        'missed_detection_rate': []
    }

    if verbose:
        print(f"\n{'SNR':<6}{'4-Class':<12}{'5-Class(All)':<14}{'5-Class(UCI)':<14}"
              f"{'DTX Det':<10}{'FA Rate':<10}")
        print("-" * 66)

    for snr in sorted(results_5class.keys()):
        r5 = results_5class[snr]
        dm = r5['dtx_metrics']

        # Get 4-class accuracy
        if snr in results_4class:
            acc_4class = float(results_4class[snr])
        else:
            acc_4class = 0.0

        comparison['snr'].append(snr)
        comparison['base_4class_accuracy'].append(acc_4class)
        comparison['dtx_5class_overall_accuracy'].append(
            float(r5['accuracy'] * 100))
        comparison['dtx_5class_uci_accuracy'].append(
            float(dm['uci_accuracy'] * 100))
        comparison['dtx_detection_rate'].append(
            float(dm['dtx_detection_rate'] * 100))
        comparison['false_alarm_rate'].append(
            float(dm['false_alarm_rate'] * 100))
        comparison['missed_detection_rate'].append(
            float(dm['missed_detection_rate'] * 100))

        if verbose:
            print(f"{snr:<6}{acc_4class:<12.2f}{r5['accuracy']*100:<14.2f}"
                  f"{dm['uci_accuracy']*100:<14.2f}{dm['dtx_detection_rate']*100:<10.2f}"
                  f"{dm['false_alarm_rate']*100:<10.4f}")

    if verbose:
        print("-" * 66)
        print("=" * 70 + "\n")

    return comparison


# =============================================================================
# SECTION 6: SAVE DTX RESULTS
# =============================================================================

def save_dtx_results(
    all_results: Dict[int, Dict],
    comparison: Dict,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Save DTX experiment results to CSV file.

    Parameters:
    -----------
    all_results : dict
        5-class evaluation results per SNR
    comparison : dict
        Comparison with 4-class model
    verbose : bool, default=True
        Print saved results

    Returns:
    --------
    df : pd.DataFrame
        DataFrame containing all results
    """

    filepath = os.path.join(config_dtx.RESULTS_DIR,
                            config_dtx.RESULTS_FILENAME)

    # Create directory
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    # Build data
    data = []
    for i, snr in enumerate(comparison['snr']):
        dm = all_results[snr]['dtx_metrics']

        row = {
            'SNR_dB': snr,
            'Base_4Class_Accuracy_pct': comparison['base_4class_accuracy'][i],
            'DTX_5Class_Overall_Accuracy_pct': comparison['dtx_5class_overall_accuracy'][i],
            'DTX_5Class_UCI_Accuracy_pct': comparison['dtx_5class_uci_accuracy'][i],
            'DTX_Detection_Rate_pct': comparison['dtx_detection_rate'][i],
            'False_Alarm_Rate_pct': comparison['false_alarm_rate'][i],
            'Missed_Detection_Rate_pct': comparison['missed_detection_rate'][i],
            '3GPP_FA_Pass': dm['false_alarm_passes_3gpp'],
            '3GPP_MD_Pass': dm['missed_detection_passes_3gpp']
        }

        # Per-class accuracy
        for c in range(config_dtx.NUM_CLASSES):
            row[f'Class_{c}_Accuracy_pct'] = float(
                all_results[snr]['per_class_accuracy'][c] * 100
            )

        data.append(row)

    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False, float_format='%.4f')

    if verbose:
        print(f"\nDTX results saved to: {filepath}")
        print("\nResults Summary:")
        print("-" * 100)
        display_cols = ['SNR_dB', 'Base_4Class_Accuracy_pct', 'DTX_5Class_Overall_Accuracy_pct',
                        'DTX_5Class_UCI_Accuracy_pct', 'DTX_Detection_Rate_pct',
                        'False_Alarm_Rate_pct', 'Missed_Detection_Rate_pct']
        print(df[display_cols].to_string(index=False))
        print("-" * 100)

    return df


def save_dtx_experiment_summary(
    all_results: Dict[int, Dict],
    comparison: Dict,
    training_time: float,
    history: Dict,
    verbose: bool = True
) -> None:
    """
    Save DTX experiment summary to text file.

    Parameters:
    -----------
    all_results : dict
        5-class evaluation results
    comparison : dict
        Comparison with 4-class model
    training_time : float
        Training time in seconds
    history : dict
        Training history
    verbose : bool, default=True
        Print confirmation
    """

    filepath = os.path.join(config_dtx.RESULTS_DIR,
                            "experiment_summary_dtx.txt")

    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    best_epoch = int(np.argmax(history['val_accuracy']) + 1)
    best_val_acc = float(max(history['val_accuracy']) * 100)

    with open(filepath, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("PUCCH Format 0 - ML Decoder with DTX Detection\n")
        f.write("Experiment Summary\n")
        f.write("=" * 70 + "\n\n")

        f.write(
            f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("--- Novel Contribution ---\n")
        f.write("Extended 4-class PUCCH Format 0 decoder to 5 classes\n")
        f.write("by adding DTX (Discontinuous Transmission) detection.\n\n")

        f.write("--- Configuration ---\n")
        f.write(f"Classes: {config_dtx.NUM_CLASSES} (4 UCI + 1 DTX)\n")
        f.write(f"Training SNR: {config_dtx.TRAIN_SNR} dB\n")
        f.write(
            f"Architecture: {config_dtx.INPUT_SIZE} -> {config_dtx.HIDDEN_LAYERS} -> {config_dtx.OUTPUT_SIZE}\n")
        f.write(f"Dropout: {config_dtx.DROPOUT_RATE}\n")
        f.write(
            f"Optimizer: SGD(lr={config_dtx.LEARNING_RATE}, momentum={config_dtx.MOMENTUM})\n")
        f.write(
            f"Training time: {training_time:.1f}s ({training_time/60:.2f} min)\n")
        f.write(
            f"Best val accuracy: {best_val_acc:.2f}% at epoch {best_epoch}\n\n")

        f.write("--- Results ---\n")
        f.write(f"{'SNR':<6}{'4-Class':<10}{'5-Class':<10}{'UCI Acc':<10}"
                f"{'DTX Det':<10}{'FA Rate':<10}{'MD Rate':<10}\n")
        f.write("-" * 66 + "\n")

        for i, snr in enumerate(comparison['snr']):
            f.write(f"{snr:<6}"
                    f"{comparison['base_4class_accuracy'][i]:<10.2f}"
                    f"{comparison['dtx_5class_overall_accuracy'][i]:<10.2f}"
                    f"{comparison['dtx_5class_uci_accuracy'][i]:<10.2f}"
                    f"{comparison['dtx_detection_rate'][i]:<10.2f}"
                    f"{comparison['false_alarm_rate'][i]:<10.4f}"
                    f"{comparison['missed_detection_rate'][i]:<10.4f}\n")

        f.write("\n" + "=" * 70 + "\n")

    if verbose:
        print(f"Experiment summary saved to: {filepath}")


# =============================================================================
# SECTION 7: DTX VISUALIZATION
# =============================================================================

def plot_dtx_metrics(
    all_results: Dict[int, Dict],
    save_path: Optional[str] = None,
    show_plot: bool = True
) -> None:
    """
    Plot DTX-specific metrics: False Alarm and Missed Detection rates.

    Parameters:
    -----------
    all_results : dict
        5-class evaluation results per SNR
    save_path : str, optional
        Path to save figure
    show_plot : bool, default=True
        Display plot
    """

    if save_path is None:
        save_path = os.path.join(config_dtx.PLOTS_DIR, "dtx_metrics.png")

    snr_values = sorted(all_results.keys())
    false_alarm = [all_results[s]['dtx_metrics']
                   ['false_alarm_rate'] * 100 for s in snr_values]
    missed_det = [all_results[s]['dtx_metrics']
                  ['missed_detection_rate'] * 100 for s in snr_values]
    dtx_det = [all_results[s]['dtx_metrics']
               ['dtx_detection_rate'] * 100 for s in snr_values]

    fig, axes = plt.subplots(1, 2, figsize=config_dtx.FIGURE_SIZE_LARGE)

    # Left: False Alarm and Missed Detection
    axes[0].plot(snr_values, false_alarm, 'r-o', linewidth=2.5, markersize=10,
                 label='False Alarm Rate', markerfacecolor='white', markeredgewidth=2)
    axes[0].plot(snr_values, missed_det, 'b-s', linewidth=2.5, markersize=10,
                 label='Missed Detection Rate', markerfacecolor='white', markeredgewidth=2)
    axes[0].axhline(y=config_dtx.FALSE_ALARM_REQUIREMENT * 100, color='green',
                    linestyle=':', linewidth=2,
                    label=f'3GPP Requirement ({config_dtx.FALSE_ALARM_REQUIREMENT*100}%)')

    axes[0].set_xlabel('SNR (dB)', fontsize=14)
    axes[0].set_ylabel('Rate (%)', fontsize=14)
    axes[0].set_title('False Alarm & Missed Detection', fontsize=16)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(snr_values)

    # Right: DTX Detection Rate
    axes[1].plot(snr_values, dtx_det, 'g-^', linewidth=2.5, markersize=10,
                 label='DTX Detection Rate', markerfacecolor='white', markeredgewidth=2)

    for snr, rate in zip(snr_values, dtx_det):
        axes[1].annotate(f'{rate:.1f}%', (snr, rate),
                         textcoords="offset points", xytext=(0, 12),
                         ha='center', fontsize=10, fontweight='bold')

    axes[1].set_xlabel('SNR (dB)', fontsize=14)
    axes[1].set_ylabel('Detection Rate (%)', fontsize=14)
    axes[1].set_title('DTX Detection Rate', fontsize=16)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(snr_values)
    axes[1].set_ylim([max(0, min(dtx_det) - 5), 102])

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


def plot_4class_vs_5class(
    comparison: Dict,
    save_path: Optional[str] = None,
    show_plot: bool = True
) -> None:
    """
    Plot comparison between 4-class and 5-class models.

    Parameters:
    -----------
    comparison : dict
        Comparison results
    save_path : str, optional
        Path to save figure
    show_plot : bool, default=True
        Display plot
    """

    if save_path is None:
        save_path = os.path.join(config_dtx.PLOTS_DIR, "4class_vs_5class.png")

    snr_values = comparison['snr']

    fig, ax = plt.subplots(figsize=config_dtx.FIGURE_SIZE_MEDIUM)

    ax.plot(snr_values, comparison['base_4class_accuracy'], 'b-o',
            linewidth=2.5, markersize=10, label='4-Class (Base)',
            markerfacecolor='white', markeredgewidth=2)
    ax.plot(snr_values, comparison['dtx_5class_uci_accuracy'], 'r--s',
            linewidth=2.5, markersize=10, label='5-Class UCI Accuracy',
            markerfacecolor='white', markeredgewidth=2)
    ax.plot(snr_values, comparison['dtx_5class_overall_accuracy'], 'g-.^',
            linewidth=2.5, markersize=10, label='5-Class Overall Accuracy',
            markerfacecolor='white', markeredgewidth=2)

    ax.axhline(y=99, color='gray', linestyle=':', linewidth=1.5,
               label='3GPP 99%', alpha=0.7)

    ax.set_xlabel('SNR (dB)', fontsize=14)
    ax.set_ylabel('Accuracy (%)', fontsize=14)
    ax.set_title('4-Class vs 5-Class (with DTX) Model Comparison', fontsize=16)
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(snr_values)

    all_acc = (comparison['base_4class_accuracy'] +
               comparison['dtx_5class_overall_accuracy'] +
               comparison['dtx_5class_uci_accuracy'])
    ax.set_ylim([max(0, min(all_acc) - 5), 102])

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


def plot_confusion_matrices_dtx(
    all_results: Dict[int, Dict],
    save_path: Optional[str] = None,
    show_plot: bool = True
) -> None:
    """
    Plot confusion matrices for 5-class model.

    This function is separate from visualization.py because that module
    uses config.NUM_CLASSES=4. This uses config_dtx.NUM_CLASSES=5.

    Parameters:
    -----------
    all_results : dict
        5-class results per SNR
    save_path : str, optional
        Path to save figure
    show_plot : bool, default=True
        Display plot
    """

    if save_path is None:
        save_path = os.path.join(config_dtx.PLOTS_DIR,
                                 "confusion_matrices_dtx.png")

    snr_values = sorted(all_results.keys())
    num_snr = len(snr_values)

    fig_width = 5 * num_snr
    fig_height = 4.5
    fig, axes = plt.subplots(1, num_snr, figsize=(fig_width, fig_height))

    if num_snr == 1:
        axes = [axes]

    # 5 class labels
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
# SECTION 8: MAIN PIPELINE
# =============================================================================

def run_dtx_pipeline(show_plots: bool = True):
    """
    Run the complete DTX detection pipeline.

    Steps:
        1. Display system info and configuration
        2. Load and merge UCI + DTX datasets
        3. Analyze training data
        4. Split train/validation
        5. Create 5-class model
        6. Train model
        7. Load best model
        8. Evaluate with DTX metrics
        9. Compare with 4-class model
        10. Generate plots
        11. Save results

    Parameters:
    -----------
    show_plots : bool, default=True
        If True, display plots during execution

    Returns:
    --------
    results : dict or None
        Dictionary containing all outputs, or None if pipeline fails
    """

    pipeline_start = time.time()

    # =========================================================================
    # STEP 1: System Info and Configuration
    # =========================================================================

    print("\n" + "#" * 70)
    print("#" + " " * 15 + "DTX DETECTION PIPELINE" + " " * 31 + "#")
    print("#" * 70)

    print(
        f"\nTimestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print_tensorflow_info()
    config_dtx.print_config()

    if not config_dtx.validate_config():
        print("ERROR: Configuration validation failed!")
        return None

    config_dtx.create_directories()
    set_random_seeds(config_dtx.MASTER_SEED)

    # =========================================================================
    # STEP 2: Load Merged Datasets
    # =========================================================================

    print("\n" + "=" * 70)
    print("STEP 1: Loading Merged Datasets (UCI + DTX)")
    print("=" * 70)

    try:
        datasets = load_all_merged_datasets(verbose=True)
    except RuntimeError as e:
        print(f"ERROR: {e}")
        return None

    if config_dtx.TRAIN_SNR not in datasets:
        print(f"ERROR: Training SNR {config_dtx.TRAIN_SNR} dB not available!")
        return None

    # =========================================================================
    # STEP 3: Analyze Training Data
    # =========================================================================

    print("\n" + "=" * 70)
    print("STEP 2: Analyzing Training Data")
    print("=" * 70)

    analyze_dataset(
        datasets[config_dtx.TRAIN_SNR][0],
        datasets[config_dtx.TRAIN_SNR][1],
        name=f"5-Class Training Data (SNR={config_dtx.TRAIN_SNR} dB)"
    )

    # =========================================================================
    # STEP 4: Prepare Data Splits
    # =========================================================================

    print("\n" + "=" * 70)
    print("STEP 3: Preparing Data Splits")
    print("=" * 70)

    X_full, y_full = datasets[config_dtx.TRAIN_SNR]

    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y_full,
        train_size=config_dtx.TRAIN_RATIO,
        random_state=config_dtx.SKLEARN_SEED,
        stratify=y_full
    )

    print(f"\nTraining: {len(X_train):,} samples")
    print(f"Validation: {len(X_val):,} samples")

    # Class distribution
    print("\nClass distribution:")
    for name, y_set in [("Train", y_train), ("Val", y_val)]:
        unique, counts = np.unique(y_set, return_counts=True)
        dist = " ".join([f"C{u}:{c:,}" for u, c in zip(unique, counts)])
        print(f"  {name}: {dist}")

    # Prepare test sets
    X_test_dict = {}
    y_test_dict = {}

    print(f"\nTest datasets:")
    for snr in sorted(datasets.keys()):
        X_test_dict[snr] = datasets[snr][0]
        y_test_dict[snr] = datasets[snr][1]
        print(f"  SNR {snr:2d} dB: {len(y_test_dict[snr]):,} samples")

    # =========================================================================
    # STEP 5: Create 5-Class Model
    # =========================================================================

    print("\n" + "=" * 70)
    print("STEP 4: Creating 5-Class Model")
    print("=" * 70)

    model = create_model(
        input_size=config_dtx.INPUT_SIZE,
        hidden_layers=config_dtx.HIDDEN_LAYERS,
        output_size=config_dtx.OUTPUT_SIZE,
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

    # =========================================================================
    # STEP 6: Train Model
    # =========================================================================

    print("\n" + "=" * 70)
    print("STEP 5: Training 5-Class Model")
    print("=" * 70)

    history, training_time = train_dtx_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        verbose=1
    )

    training_summary = get_training_summary(history)

    # =========================================================================
    # STEP 7: Load Best Model
    # =========================================================================

    print("\n" + "=" * 70)
    print("STEP 6: Loading Best Model")
    print("=" * 70)

    model_filepath = config_dtx.get_model_filepath()

    try:
        best_model = load_saved_model(model_filepath)
    except FileNotFoundError:
        print("WARNING: Best model file not found. Using last training state.")
        best_model = model

    # =========================================================================
    # STEP 8: Evaluate with DTX Metrics
    # =========================================================================

    print("\n" + "=" * 70)
    print("STEP 7: Evaluating 5-Class Model")
    print("=" * 70)

    all_results = evaluate_dtx_all_snr(
        model=best_model,
        X_test_dict=X_test_dict,
        y_test_dict=y_test_dict,
        verbose=True
    )

    # =========================================================================
    # STEP 9: Compare with 4-Class Model
    # =========================================================================

    print("\n" + "=" * 70)
    print("STEP 8: Comparing with 4-Class Base Model")
    print("=" * 70)

    # Load 4-class results from previous experiment
    base_results_path = os.path.join("./results/", "results_summary.csv")
    results_4class = {}

    if os.path.exists(base_results_path):
        base_df = pd.read_csv(base_results_path)
        for _, row in base_df.iterrows():
            results_4class[int(row['SNR_dB'])] = float(row['NN_Accuracy_pct'])
        print(f"Loaded 4-class results from: {base_results_path}")
    else:
        print(f"WARNING: 4-class results not found at: {base_results_path}")
        print("Using zeros for comparison. Run main.py first for full comparison.")
        for snr in config_dtx.SNR_VALUES:
            results_4class[snr] = 0.0

    comparison = compare_4class_vs_5class(
        results_4class=results_4class,
        results_5class=all_results,
        verbose=True
    )

    # =========================================================================
    # STEP 10: Generate Plots
    # =========================================================================

    print("\n" + "=" * 70)
    print("STEP 9: Generating Plots")
    print("=" * 70)

    # Training history
    print("\n1. Training history...")
    plot_training_history(
        history=history,
        save_path=os.path.join(config_dtx.PLOTS_DIR,
                               "training_history_dtx.png"),
        show_plot=show_plots
    )

    # Confusion matrices (5 classes)
    print("2. Confusion matrices (5 classes)...")
    plot_confusion_matrices_dtx(
        all_results=all_results,
        save_path=os.path.join(config_dtx.PLOTS_DIR,
                               "confusion_matrices_dtx.png"),
        show_plot=show_plots
    )

    # DTX-specific metrics
    print("3. DTX metrics...")
    plot_dtx_metrics(
        all_results=all_results,
        save_path=os.path.join(config_dtx.PLOTS_DIR, "dtx_metrics.png"),
        show_plot=show_plots
    )

    # 4-class vs 5-class comparison
    print("4. 4-class vs 5-class comparison...")
    plot_4class_vs_5class(
        comparison=comparison,
        save_path=os.path.join(config_dtx.PLOTS_DIR, "4class_vs_5class.png"),
        show_plot=show_plots
    )

    print(f"\nAll plots saved to: {config_dtx.PLOTS_DIR}")

    # =========================================================================
    # STEP 11: Save Results
    # =========================================================================

    print("\n" + "=" * 70)
    print("STEP 10: Saving Results")
    print("=" * 70)

    # Save CSV results
    results_df = save_dtx_results(
        all_results=all_results,
        comparison=comparison,
        verbose=True
    )

    # Save experiment summary
    save_dtx_experiment_summary(
        all_results=all_results,
        comparison=comparison,
        training_time=training_time,
        history=history,
        verbose=True
    )

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================

    pipeline_time = time.time() - pipeline_start

    print("\n" + "#" * 70)
    print("#" + " " * 15 + "DTX PIPELINE COMPLETE" + " " * 32 + "#")
    print("#" * 70)

    print(
        f"\nTotal pipeline time: {pipeline_time:.1f}s ({pipeline_time/60:.2f} min)")
    print(f"Training time: {training_time:.1f}s ({training_time/60:.2f} min)")
    print(f"Best val accuracy: {training_summary['best_val_acc']:.2f}% "
          f"(epoch {training_summary['best_epoch']})")

    print(f"\n--- DTX Results Summary ---")
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
    print(f"Model: {model_filepath}")
    print(
        f"Results: {os.path.join(config_dtx.RESULTS_DIR, config_dtx.RESULTS_FILENAME)}")
    print(
        f"Summary: {os.path.join(config_dtx.RESULTS_DIR, 'experiment_summary_dtx.txt')}")
    print(f"Plots: {config_dtx.PLOTS_DIR}")
    print(f"Logs: {config_dtx.LOGS_DIR}")

    print("\n" + "#" * 70 + "\n")

    return {
        'model': best_model,
        'history': history,
        'training_time': training_time,
        'training_summary': training_summary,
        'all_results': all_results,
        'comparison': comparison,
        'results_df': results_df,
        'pipeline_time': pipeline_time
    }


# =============================================================================
# SECTION 9: ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    """
    Main entry point for DTX detection experiment.

    Usage:
        python main_dtx.py              Run with plots
        python main_dtx.py --no-plots   Run without displaying plots
        python main_dtx.py --help       Show help
    """

    show_plots = True

    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg == "--no-plots":
                show_plots = False
            elif arg == "--help":
                print("\nUsage: python main_dtx.py [options]")
                print("\nOptions:")
                print("  --no-plots    Run without displaying plots")
                print("  --help        Show this help message")
                print("\nThis script trains and evaluates a 5-class PUCCH Format 0")
                print("decoder that includes DTX detection as a novel contribution.")
                sys.exit(0)
            else:
                print(f"Unknown argument: {arg}")
                print("Use --help for usage information")
                sys.exit(1)

    results = run_dtx_pipeline(show_plots=show_plots)

    if results is not None:
        print("DTX pipeline completed successfully!")
    else:
        print("DTX pipeline failed! Check error messages above.")
        sys.exit(1)
