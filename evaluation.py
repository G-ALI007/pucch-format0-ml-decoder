"""
================================================================================
PUCCH Format 0 - Machine Learning Decoder
Evaluation Module
================================================================================

This module handles model evaluation and comparison including:
    - Evaluating neural network on single and multiple SNR values
    - Computing accuracy, precision, recall, F1-score
    - Generating confusion matrices
    - Implementing correlation-based decoder for comparison
    - Comparing neural network vs correlation decoder
    - Generating classification reports
    - Saving evaluation results

================================================================================
"""

import os
import time
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional, Any

# Scikit-learn metrics
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)

# TensorFlow
from tf_keras.models import Sequential

# Import project modules
from config import config
from data_loader import reconstruct_complex


# =============================================================================
# SECTION 1: SINGLE SNR EVALUATION
# =============================================================================

def evaluate_single_snr(
    model: Sequential,
    X_test: np.ndarray,
    y_test: np.ndarray,
    snr_db: int,
    verbose: bool = True
) -> Dict:
    """
    Evaluate the neural network model on a single SNR test set.

    This function computes:
        - Overall accuracy
        - Per-class accuracy
        - Precision, recall, F1-score (weighted)
        - Confusion matrix
        - Number of correct and incorrect predictions

    Parameters:
    -----------
    model : keras.Sequential
        Trained neural network model
    X_test : np.ndarray
        Test features, shape (num_samples, 24)
    y_test : np.ndarray
        True test labels, shape (num_samples,)
    snr_db : int
        SNR value in dB (for labeling purposes)
    verbose : bool, default=True
        If True, print evaluation results

    Returns:
    --------
    results : dict
        Dictionary containing:
        - 'snr_db': SNR value
        - 'accuracy': Overall accuracy (0 to 1)
        - 'precision': Weighted precision (0 to 1)
        - 'recall': Weighted recall (0 to 1)
        - 'f1_score': Weighted F1-score (0 to 1)
        - 'per_class_accuracy': Array of per-class accuracies
        - 'confusion_matrix': Confusion matrix array
        - 'y_true': True labels
        - 'y_pred': Predicted labels
        - 'y_pred_proba': Predicted probabilities
        - 'num_samples': Total number of samples
        - 'num_correct': Number of correct predictions
        - 'num_errors': Number of incorrect predictions

    Example:
    --------
    >>> results = evaluate_single_snr(model, X_test, y_test, snr_db=10)
    SNR =  10 dB: Accuracy =  98.45% (196900/200000)
    """

    # Validate inputs
    if X_test.shape[0] == 0:
        raise ValueError("Test data is empty")
    if X_test.shape[0] != y_test.shape[0]:
        raise ValueError(
            f"X_test and y_test have different number of samples: "
            f"{X_test.shape[0]} vs {y_test.shape[0]}"
        )

    # Get predictions
    y_pred_proba = model.predict(X_test, batch_size=256, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)

    # Calculate overall metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(
        y_test, y_pred, labels=list(range(config.NUM_CLASSES)))

    # Calculate per-class accuracy
    per_class_acc = np.zeros(config.NUM_CLASSES)
    for c in range(config.NUM_CLASSES):
        class_mask = (y_test == c)
        if np.sum(class_mask) > 0:
            per_class_acc[c] = accuracy_score(
                y_test[class_mask], y_pred[class_mask])
        else:
            per_class_acc[c] = 0.0

    # Count correct and incorrect
    num_correct = int(np.sum(y_test == y_pred))
    num_errors = int(np.sum(y_test != y_pred))
    num_samples = int(len(y_test))

    # Store results
    results = {
        'snr_db': snr_db,
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'per_class_accuracy': per_class_acc,
        'confusion_matrix': conf_matrix,
        'y_true': y_test,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'num_samples': num_samples,
        'num_correct': num_correct,
        'num_errors': num_errors
    }

    # Print results
    if verbose:
        print(f"SNR = {snr_db:3d} dB: Accuracy = {accuracy*100:6.2f}% "
              f"({num_correct:,}/{num_samples:,})")

    return results


# =============================================================================
# SECTION 2: ALL SNR EVALUATION
# =============================================================================

def evaluate_all_snr(
    model: Sequential,
    X_test_dict: Dict[int, np.ndarray],
    y_test_dict: Dict[int, np.ndarray],
    verbose: bool = True
) -> Dict[int, Dict]:
    """
    Evaluate the neural network model on all SNR test sets.

    Parameters:
    -----------
    model : keras.Sequential
        Trained neural network model
    X_test_dict : dict
        Dictionary mapping SNR to test features
        Key: int (SNR in dB)
        Value: np.ndarray of shape (num_samples, 24)
    y_test_dict : dict
        Dictionary mapping SNR to test labels
        Key: int (SNR in dB)
        Value: np.ndarray of shape (num_samples,)
    verbose : bool, default=True
        If True, print detailed results

    Returns:
    --------
    all_results : dict
        Dictionary mapping SNR to evaluation results
        Key: int (SNR in dB)
        Value: dict (as returned by evaluate_single_snr)

    Example:
    --------
    >>> all_results = evaluate_all_snr(model, X_test_dict, y_test_dict)
    ======================================================================
    EVALUATING MODEL ON ALL SNR VALUES
    ======================================================================
    SNR =   0 dB: Accuracy =  84.23% (168460/200000)
    SNR =   5 dB: Accuracy =  95.67% (191340/200000)
    SNR =  10 dB: Accuracy =  98.45% (196900/200000)
    SNR =  15 dB: Accuracy =  99.34% (198680/200000)
    SNR =  20 dB: Accuracy =  99.78% (199560/200000)
    ...
    """

    if verbose:
        print("\n" + "=" * 70)
        print("EVALUATING MODEL ON ALL SNR VALUES")
        print("=" * 70 + "\n")

    all_results = {}

    # Evaluate each SNR
    for snr in sorted(X_test_dict.keys()):
        X_test = X_test_dict[snr]
        y_test = y_test_dict[snr]

        results = evaluate_single_snr(
            model=model,
            X_test=X_test,
            y_test=y_test,
            snr_db=snr,
            verbose=verbose
        )

        all_results[snr] = results

    # Print summary table
    if verbose:
        print("\n" + "-" * 80)
        print(f"{'SNR (dB)':<10}{'Accuracy':<12}{'Precision':<12}"
              f"{'Recall':<12}{'F1-Score':<12}{'Errors':<12}")
        print("-" * 80)

        for snr in sorted(all_results.keys()):
            r = all_results[snr]
            print(f"{snr:<10}"
                  f"{r['accuracy']*100:<12.2f}"
                  f"{r['precision']*100:<12.2f}"
                  f"{r['recall']*100:<12.2f}"
                  f"{r['f1_score']*100:<12.2f}"
                  f"{r['num_errors']:<12,}")

        print("-" * 80)

        # Print per-class accuracy
        print(f"\nPer-Class Accuracy (%):")
        header = f"{'SNR (dB)':<10}"
        for c in range(config.NUM_CLASSES):
            header += f"{'Class '+str(c):<12}"
        print(header)
        print("-" * (10 + 12 * config.NUM_CLASSES))

        for snr in sorted(all_results.keys()):
            row = f"{snr:<10}"
            for c in range(config.NUM_CLASSES):
                acc = all_results[snr]['per_class_accuracy'][c] * 100
                row += f"{acc:<12.2f}"
            print(row)

        print("-" * (10 + 12 * config.NUM_CLASSES))

    if verbose:
        print("=" * 70 + "\n")

    return all_results


# =============================================================================
# SECTION 3: CLASSIFICATION REPORTS
# =============================================================================

def generate_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    snr_db: int,
    verbose: bool = True
) -> str:
    """
    Generate a detailed classification report for one SNR value.

    Parameters:
    -----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    snr_db : int
        SNR value in dB
    verbose : bool, default=True
        If True, print the report

    Returns:
    --------
    report : str
        Classification report as string
    """

    target_names = [f"Class {i}" for i in range(config.NUM_CLASSES)]

    report = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        digits=4,
        zero_division=0
    )

    if verbose:
        print(f"\n--- Classification Report: SNR = {snr_db} dB ---")
        print(report)

    return report


def generate_all_classification_reports(
    all_results: Dict[int, Dict],
    verbose: bool = True
) -> Dict[int, str]:
    """
    Generate classification reports for all SNR values.

    Parameters:
    -----------
    all_results : dict
        Results from evaluate_all_snr()
    verbose : bool, default=True
        If True, print all reports

    Returns:
    --------
    reports : dict
        Dictionary mapping SNR to classification report string
    """

    if verbose:
        print("\n" + "=" * 70)
        print("CLASSIFICATION REPORTS")
        print("=" * 70)

    reports = {}

    for snr in sorted(all_results.keys()):
        y_true = all_results[snr]['y_true']
        y_pred = all_results[snr]['y_pred']

        report = generate_classification_report(
            y_true=y_true,
            y_pred=y_pred,
            snr_db=snr,
            verbose=verbose
        )

        reports[snr] = report

    if verbose:
        print("=" * 70 + "\n")

    return reports


# =============================================================================
# SECTION 4: CORRELATION-BASED DECODER
# =============================================================================

def compute_reference_signals(
    X_train: np.ndarray,
    y_train: np.ndarray,
    verbose: bool = True
) -> np.ndarray:
    """
    Compute reference signals for correlation-based decoder.

    Reference signals are computed as the mean of all training samples
    for each class in the complex domain.

    Parameters:
    -----------
    X_train : np.ndarray
        Training features, shape (num_samples, 24)
    y_train : np.ndarray
        Training labels, shape (num_samples,)
    verbose : bool, default=True
        If True, print information

    Returns:
    --------
    reference_signals : np.ndarray
        Complex reference signals, shape (num_classes, 12)
        dtype: complex64
    """

    # Reconstruct complex samples from training data
    X_train_complex = reconstruct_complex(X_train)

    # Compute mean for each class
    reference_signals = np.zeros(
        (config.NUM_CLASSES, config.NUM_SUBCARRIERS),
        dtype=np.complex64
    )

    for c in range(config.NUM_CLASSES):
        class_mask = (y_train == c)
        class_count = np.sum(class_mask)

        if class_count > 0:
            reference_signals[c] = np.mean(X_train_complex[class_mask], axis=0)

        if verbose:
            print(f"  Class {c}: {class_count:,} samples used for reference")

    return reference_signals


def correlation_decode_single(
    sample_complex: np.ndarray,
    reference_signals: np.ndarray
) -> Tuple[int, np.ndarray]:
    """
    Decode a single sample using normalized correlation.

    Parameters:
    -----------
    sample_complex : np.ndarray
        Single complex sample, shape (12,)
    reference_signals : np.ndarray
        Reference signals, shape (num_classes, 12)

    Returns:
    --------
    predicted_class : int
        Predicted class label
    correlations : np.ndarray
        Correlation values for each class
    """

    num_classes = reference_signals.shape[0]
    correlations = np.zeros(num_classes, dtype=np.float64)

    # Energy of received sample
    sample_energy = np.sqrt(np.sum(np.abs(sample_complex) ** 2))

    for c in range(num_classes):
        ref = reference_signals[c]

        # Compute normalized correlation
        # correlation = |sum(sample * conj(ref))| / (||sample|| * ||ref||)
        correlation_value = np.abs(np.sum(sample_complex * np.conj(ref)))
        ref_energy = np.sqrt(np.sum(np.abs(ref) ** 2))

        # Avoid division by zero
        denominator = sample_energy * ref_energy
        if denominator > 0:
            correlations[c] = correlation_value / denominator
        else:
            correlations[c] = 0.0

    predicted_class = int(np.argmax(correlations))

    return predicted_class, correlations


def correlation_decode_batch(
    X_test: np.ndarray,
    y_test: np.ndarray,
    reference_signals: np.ndarray,
    verbose: bool = False
) -> Dict:
    """
    Decode a batch of samples using correlation-based decoder.

    Parameters:
    -----------
    X_test : np.ndarray
        Test features, shape (num_samples, 24)
    y_test : np.ndarray
        True test labels, shape (num_samples,)
    reference_signals : np.ndarray
        Complex reference signals, shape (num_classes, 12)
    verbose : bool, default=False
        If True, print progress

    Returns:
    --------
    results : dict
        Dictionary containing:
        - 'accuracy': Overall accuracy
        - 'y_pred': Predicted labels
        - 'confusion_matrix': Confusion matrix
        - 'num_correct': Number of correct predictions
        - 'num_errors': Number of incorrect predictions
        - 'num_samples': Total samples
    """

    # Reconstruct complex samples
    X_test_complex = reconstruct_complex(X_test)

    num_samples = len(X_test_complex)
    y_pred = np.zeros(num_samples, dtype=np.int32)

    # Decode each sample
    for i in range(num_samples):
        predicted_class, _ = correlation_decode_single(
            sample_complex=X_test_complex[i],
            reference_signals=reference_signals
        )
        y_pred[i] = predicted_class

        # Print progress
        if verbose and (i + 1) % 50000 == 0:
            print(f"  Decoded {i+1:,}/{num_samples:,} samples")

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(
        y_test, y_pred, labels=list(range(config.NUM_CLASSES)))
    num_correct = int(np.sum(y_test == y_pred))
    num_errors = int(np.sum(y_test != y_pred))

    results = {
        'accuracy': float(accuracy),
        'y_pred': y_pred,
        'confusion_matrix': conf_matrix,
        'num_correct': num_correct,
        'num_errors': num_errors,
        'num_samples': num_samples
    }

    return results


# =============================================================================
# SECTION 5: COMPARISON NN vs CORRELATION
# =============================================================================

def compare_nn_vs_correlation(
    model: Sequential,
    X_test_dict: Dict[int, np.ndarray],
    y_test_dict: Dict[int, np.ndarray],
    X_train: np.ndarray,
    y_train: np.ndarray,
    verbose: bool = True
) -> Dict:
    """
    Compare neural network with correlation-based decoder across all SNR values.

    Parameters:
    -----------
    model : keras.Sequential
        Trained neural network model
    X_test_dict : dict
        Test features for each SNR
    y_test_dict : dict
        Test labels for each SNR
    X_train : np.ndarray
        Training features (for computing reference signals)
    y_train : np.ndarray
        Training labels
    verbose : bool, default=True
        If True, print comparison results

    Returns:
    --------
    comparison : dict
        Dictionary containing:
        - 'snr': List of SNR values
        - 'nn_accuracy': List of NN accuracies (%)
        - 'corr_accuracy': List of correlation accuracies (%)
        - 'nn_results': Dict of NN results per SNR
        - 'corr_results': Dict of correlation results per SNR
    """

    if verbose:
        print("\n" + "=" * 70)
        print("COMPARING NEURAL NETWORK vs CORRELATION DECODER")
        print("=" * 70)

    # Compute reference signals from training data
    if verbose:
        print("\nComputing reference signals from training data:")

    reference_signals = compute_reference_signals(
        X_train=X_train,
        y_train=y_train,
        verbose=verbose
    )

    # Initialize comparison storage
    comparison = {
        'snr': [],
        'nn_accuracy': [],
        'corr_accuracy': [],
        'nn_results': {},
        'corr_results': {}
    }

    if verbose:
        print(
            f"\n{'SNR (dB)':<10}{'NN Accuracy':<15}{'Corr Accuracy':<15}{'Gain':<10}")
        print("-" * 50)

    # Compare for each SNR
    for snr in sorted(X_test_dict.keys()):
        X_test = X_test_dict[snr]
        y_test = y_test_dict[snr]

        # Neural network prediction
        y_pred_nn = np.argmax(
            model.predict(X_test, batch_size=256, verbose=0),
            axis=1
        )
        nn_acc = accuracy_score(y_test, y_pred_nn) * 100

        # Correlation-based prediction
        corr_results = correlation_decode_batch(
            X_test=X_test,
            y_test=y_test,
            reference_signals=reference_signals,
            verbose=False
        )
        corr_acc = corr_results['accuracy'] * 100

        # Calculate gain
        gain = nn_acc - corr_acc

        # Store results
        comparison['snr'].append(snr)
        comparison['nn_accuracy'].append(float(nn_acc))
        comparison['corr_accuracy'].append(float(corr_acc))
        comparison['nn_results'][snr] = {
            'y_pred': y_pred_nn, 'accuracy': float(nn_acc)}
        comparison['corr_results'][snr] = corr_results

        # Print
        if verbose:
            print(f"{snr:<10}{nn_acc:<15.2f}{corr_acc:<15.2f}{gain:<+10.2f}")

    # Print summary
    if verbose:
        print("-" * 50)

        avg_nn = float(np.mean(comparison['nn_accuracy']))
        avg_corr = float(np.mean(comparison['corr_accuracy']))
        avg_gain = avg_nn - avg_corr

        print(f"{'Average':<10}{avg_nn:<15.2f}{avg_corr:<15.2f}{avg_gain:<+10.2f}")
        print("=" * 70 + "\n")

    return comparison


# =============================================================================
# SECTION 6: RESULTS SAVING
# =============================================================================

def save_evaluation_results(
    all_results: Dict[int, Dict],
    comparison: Dict,
    filepath: Optional[str] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Save evaluation results to CSV file.

    Parameters:
    -----------
    all_results : dict
        Neural network results from evaluate_all_snr()
    comparison : dict
        Comparison results from compare_nn_vs_correlation()
    filepath : str, optional
        Path to save the CSV file
        Default: config.RESULTS_DIR + config.RESULTS_FILENAME
    verbose : bool, default=True
        If True, print saved results

    Returns:
    --------
    df : pd.DataFrame
        DataFrame containing all results
    """

    if filepath is None:
        filepath = os.path.join(config.RESULTS_DIR, config.RESULTS_FILENAME)

    # Create directory if needed
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    # Build results table
    data = []

    for i, snr in enumerate(comparison['snr']):
        row = {
            'SNR_dB': snr,
            'NN_Accuracy_pct': comparison['nn_accuracy'][i],
            'Correlation_Accuracy_pct': comparison['corr_accuracy'][i],
            'Gain_pct': comparison['nn_accuracy'][i] - comparison['corr_accuracy'][i],
            'NN_Precision_pct': all_results[snr]['precision'] * 100,
            'NN_Recall_pct': all_results[snr]['recall'] * 100,
            'NN_F1_Score_pct': all_results[snr]['f1_score'] * 100,
            'Num_Samples': all_results[snr]['num_samples'],
            'Num_Errors': all_results[snr]['num_errors']
        }

        # Add per-class accuracy
        for c in range(config.NUM_CLASSES):
            row[f'Class_{c}_Accuracy_pct'] = float(
                all_results[snr]['per_class_accuracy'][c] * 100
            )

        data.append(row)

    # Create DataFrame
    df = pd.DataFrame(data)

    # Save to CSV
    df.to_csv(filepath, index=False, float_format='%.4f')

    if verbose:
        print(f"\nResults saved to: {filepath}")
        print("\nResults Summary:")
        print("-" * 90)
        # Print subset of columns for readability
        display_cols = ['SNR_dB', 'NN_Accuracy_pct', 'Correlation_Accuracy_pct',
                        'Gain_pct', 'Num_Errors']
        print(df[display_cols].to_string(index=False))
        print("-" * 90)

    return df


def save_confusion_matrices(
    all_results: Dict[int, Dict],
    filepath: Optional[str] = None,
    verbose: bool = True
) -> None:
    """
    Save confusion matrices for all SNR values to a text file.

    Parameters:
    -----------
    all_results : dict
        Results from evaluate_all_snr()
    filepath : str, optional
        Path to save the file
    verbose : bool, default=True
        If True, print confirmation
    """

    if filepath is None:
        filepath = os.path.join(config.RESULTS_DIR, "confusion_matrices.txt")

    # Create directory if needed
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    with open(filepath, 'w') as f:
        f.write("PUCCH Format 0 - Confusion Matrices\n")
        f.write("=" * 50 + "\n\n")

        for snr in sorted(all_results.keys()):
            conf_matrix = all_results[snr]['confusion_matrix']
            accuracy = all_results[snr]['accuracy'] * 100

            f.write(f"SNR = {snr} dB (Accuracy = {accuracy:.2f}%)\n")
            f.write("-" * 40 + "\n")

            # Header
            header = "Actual\\Pred"
            for c in range(config.NUM_CLASSES):
                header += f"\tC{c}"
            f.write(header + "\n")

            # Rows
            for r in range(config.NUM_CLASSES):
                row = f"C{r}"
                for c_val in range(config.NUM_CLASSES):
                    row += f"\t{conf_matrix[r, c_val]}"
                f.write(row + "\n")

            f.write("\n")

    if verbose:
        print(f"Confusion matrices saved to: {filepath}")


def save_experiment_summary(
    all_results: Dict[int, Dict],
    comparison: Dict,
    training_time: float,
    history: Dict,
    filepath: Optional[str] = None,
    verbose: bool = True
) -> None:
    """
    Save a complete experiment summary to text file.

    Parameters:
    -----------
    all_results : dict
        Neural network results
    comparison : dict
        Comparison results
    training_time : float
        Total training time in seconds
    history : dict
        Training history
    filepath : str, optional
        Path to save the summary
    verbose : bool, default=True
        If True, print confirmation
    """

    if filepath is None:
        filepath = os.path.join(config.RESULTS_DIR, "experiment_summary.txt")

    # Create directory if needed
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    # Get training summary
    best_epoch = int(np.argmax(history['val_accuracy']) + 1)
    best_val_acc = float(max(history['val_accuracy']) * 100)

    with open(filepath, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("PUCCH Format 0 - ML Decoder Experiment Summary\n")
        f.write("=" * 70 + "\n\n")

        # Configuration
        f.write("--- Experiment Configuration ---\n")
        f.write(f"Training SNR: {config.TRAIN_SNR} dB\n")
        f.write(f"SNR values tested: {config.SNR_VALUES}\n")
        f.write(
            f"Train/Val split: {config.TRAIN_RATIO}/{config.VALIDATION_RATIO}\n")
        f.write(f"Normalize features: {config.NORMALIZE_FEATURES}\n\n")

        # Architecture
        f.write("--- Neural Network Architecture ---\n")
        f.write(f"Input size: {config.INPUT_SIZE}\n")
        f.write(f"Hidden layers: {config.HIDDEN_LAYERS}\n")
        f.write(f"Hidden activation: {config.HIDDEN_ACTIVATION}\n")
        f.write(f"Dropout rate: {config.DROPOUT_RATE}\n")
        f.write(f"Output size: {config.OUTPUT_SIZE}\n")
        f.write(f"Output activation: {config.OUTPUT_ACTIVATION}\n\n")

        # Training
        f.write("--- Training Parameters ---\n")
        f.write(f"Epochs: {config.NUM_EPOCHS}\n")
        f.write(f"Batch size: {config.BATCH_SIZE}\n")
        f.write(
            f"Optimizer: SGD (lr={config.LEARNING_RATE}, momentum={config.MOMENTUM})\n")
        f.write(f"Loss: {config.LOSS_FUNCTION}\n")
        f.write(
            f"Training time: {training_time:.2f} seconds ({training_time/60:.2f} min)\n")
        f.write(f"Epochs completed: {len(history['loss'])}\n")
        f.write(
            f"Best validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch}\n\n")

        # Results
        f.write("--- Results ---\n")
        f.write(
            f"{'SNR (dB)':<10}{'NN Acc (%)':<12}{'Corr Acc (%)':<14}{'Gain (%)':<10}\n")
        f.write("-" * 46 + "\n")

        for i, snr in enumerate(comparison['snr']):
            nn_acc = comparison['nn_accuracy'][i]
            corr_acc = comparison['corr_accuracy'][i]
            gain = nn_acc - corr_acc
            f.write(f"{snr:<10}{nn_acc:<12.2f}{corr_acc:<14.2f}{gain:<+10.2f}\n")

        f.write("\n" + "=" * 70 + "\n")

    if verbose:
        print(f"Experiment summary saved to: {filepath}")


# =============================================================================
# SECTION 7: SELF-TEST
# =============================================================================

if __name__ == "__main__":
    """
    Self-test for evaluation module.
    """

    print("\n" + "=" * 70)
    print("EVALUATION MODULE - SELF TEST")
    print("=" * 70)

    # Create dummy model and data
    print("\n--- Creating dummy model and data ---")

    np.random.seed(42)

    # Create dummy model
    from model import create_model, set_random_seeds
    set_random_seeds(42, verbose=False)

    dummy_model = create_model(print_summary=False)

    # Create dummy data
    num_samples = 500
    X_dummy = np.random.randn(num_samples, 24).astype(np.float32)
    y_dummy = np.random.randint(0, 4, num_samples).astype(np.int32)

    # Training data for reference signals
    X_train_dummy = np.random.randn(1000, 24).astype(np.float32)
    y_train_dummy = np.repeat(np.arange(4), 250).astype(np.int32)

    print(f"Dummy model created")
    print(f"Test data: X={X_dummy.shape}, y={y_dummy.shape}")
    print(f"Train data: X={X_train_dummy.shape}, y={y_train_dummy.shape}")

    # Test 1: evaluate_single_snr
    print("\n--- Test 1: evaluate_single_snr ---")
    try:
        results = evaluate_single_snr(
            model=dummy_model,
            X_test=X_dummy,
            y_test=y_dummy,
            snr_db=10,
            verbose=True
        )
        print(f"Keys: {list(results.keys())}")
        print(f"Accuracy: {results['accuracy']*100:.2f}%")
        print("Test 1: PASS")
    except Exception as e:
        print(f"Test 1: FAIL - {e}")

    # Test 2: evaluate_all_snr
    print("\n--- Test 2: evaluate_all_snr ---")
    try:
        X_test_dict = {0: X_dummy.copy(), 10: X_dummy.copy(),
                       20: X_dummy.copy()}
        y_test_dict = {0: y_dummy.copy(), 10: y_dummy.copy(),
                       20: y_dummy.copy()}

        all_results = evaluate_all_snr(
            model=dummy_model,
            X_test_dict=X_test_dict,
            y_test_dict=y_test_dict,
            verbose=True
        )
        print(f"SNR values evaluated: {list(all_results.keys())}")
        print("Test 2: PASS")
    except Exception as e:
        print(f"Test 2: FAIL - {e}")

    # Test 3: generate_classification_report
    print("\n--- Test 3: generate_classification_report ---")
    try:
        report = generate_classification_report(
            y_true=y_dummy,
            y_pred=results['y_pred'],
            snr_db=10,
            verbose=True
        )
        print("Test 3: PASS")
    except Exception as e:
        print(f"Test 3: FAIL - {e}")

    # Test 4: generate_all_classification_reports
    print("\n--- Test 4: generate_all_classification_reports ---")
    try:
        reports = generate_all_classification_reports(
            all_results=all_results,
            verbose=False  # Don't print all reports in test
        )
        print(f"Reports generated for SNR: {list(reports.keys())}")
        print("Test 4: PASS")
    except Exception as e:
        print(f"Test 4: FAIL - {e}")

    # Test 5: compute_reference_signals
    print("\n--- Test 5: compute_reference_signals ---")
    try:
        ref_signals = compute_reference_signals(
            X_train=X_train_dummy,
            y_train=y_train_dummy,
            verbose=True
        )
        print(f"Reference signals shape: {ref_signals.shape}")
        print(f"Reference signals dtype: {ref_signals.dtype}")
        print("Test 5: PASS")
    except Exception as e:
        print(f"Test 5: FAIL - {e}")

    # Test 6: correlation_decode_single
    print("\n--- Test 6: correlation_decode_single ---")
    try:
        sample_complex = reconstruct_complex(X_dummy[:1])[0]

        pred_class, correlations = correlation_decode_single(
            sample_complex=sample_complex,
            reference_signals=ref_signals
        )
        print(f"Predicted class: {pred_class}")
        print(f"Correlations: {correlations}")
        print("Test 6: PASS")
    except Exception as e:
        print(f"Test 6: FAIL - {e}")

    # Test 7: correlation_decode_batch
    print("\n--- Test 7: correlation_decode_batch ---")
    try:
        corr_results = correlation_decode_batch(
            X_test=X_dummy[:100],
            y_test=y_dummy[:100],
            reference_signals=ref_signals,
            verbose=False
        )
        print(f"Correlation accuracy: {corr_results['accuracy']*100:.2f}%")
        print("Test 7: PASS")
    except Exception as e:
        print(f"Test 7: FAIL - {e}")

    # Test 8: compare_nn_vs_correlation
    print("\n--- Test 8: compare_nn_vs_correlation ---")
    try:
        comparison = compare_nn_vs_correlation(
            model=dummy_model,
            X_test_dict=X_test_dict,
            y_test_dict=y_test_dict,
            X_train=X_train_dummy,
            y_train=y_train_dummy,
            verbose=True
        )
        print(f"Comparison SNR values: {comparison['snr']}")
        print("Test 8: PASS")
    except Exception as e:
        print(f"Test 8: FAIL - {e}")

    # Test 9: save_evaluation_results
    print("\n--- Test 9: save_evaluation_results ---")
    try:
        test_dir = "./test_temp_eval/"
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)

        test_path = os.path.join(test_dir, "test_results.csv")

        df = save_evaluation_results(
            all_results=all_results,
            comparison=comparison,
            filepath=test_path,
            verbose=True
        )
        print(f"DataFrame shape: {df.shape}")

        # Cleanup
        if os.path.exists(test_path):
            os.remove(test_path)
        if os.path.exists(test_dir):
            os.rmdir(test_dir)

        print("Test 9: PASS")
    except Exception as e:
        print(f"Test 9: FAIL - {e}")

    # Test 10: save_confusion_matrices
    print("\n--- Test 10: save_confusion_matrices ---")
    try:
        test_dir = "./test_temp_eval/"
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)

        test_path = os.path.join(test_dir, "test_cm.txt")

        save_confusion_matrices(
            all_results=all_results,
            filepath=test_path,
            verbose=True
        )

        # Verify file exists and has content
        if os.path.exists(test_path):
            file_size = os.path.getsize(test_path)
            print(f"File size: {file_size} bytes")

        # Cleanup
        if os.path.exists(test_path):
            os.remove(test_path)
        if os.path.exists(test_dir):
            os.rmdir(test_dir)

        print("Test 10: PASS")
    except Exception as e:
        print(f"Test 10: FAIL - {e}")

    print("\n" + "=" * 70)
    print("SELF TEST COMPLETE")
    print("=" * 70)
