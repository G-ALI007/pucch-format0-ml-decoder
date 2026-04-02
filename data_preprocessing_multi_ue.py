"""
PUCCH Format 0 - Multi-User Machine Learning Decoder
Data Preprocessing Module

This module handles data preprocessing for Multi-User scenarios including:
- Splitting data into training and validation sets
- Feature normalization (optional)
- Data preparation for neural network training

Functions:
- split_train_validation: Split data into train/val sets
- prepare_data: Prepare all data splits for training
- normalize_features: Normalize features using StandardScaler or MinMaxScaler
- preprocess_pipeline: Complete preprocessing pipeline
- load_scaler: Load a saved scaler from disk
- save_scaler: Save a scaler to disk

Usage:
    from data_preprocessing_multi_ue import preprocess_pipeline
    X_train, X_val, X_test_dict, scaler = preprocess_pipeline(...)
================================================================================
"""
import os
import pickle
import numpy as np
from typing import Dict, Tuple, Optional, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Import configuration
from config_multi_ue import config_multi_ue


# =============================================================================
# SECTION 1: DATA SPLITTING
# =============================================================================

def split_train_validation(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: Optional[float] = None,
    random_seed: Optional[int] = None,
    stratify: Optional[bool] = None,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into training and validation sets.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix of shape (num_samples, num_features)
    y : np.ndarray
        Label vector of shape (num_samples,)
    train_ratio : float, optional
        Fraction of data for training (e.g., 0.75 for 75%)
        If None, uses config_multi_ue.TRAIN_RATIO
    random_seed : int, optional
        Random seed for reproducibility
        If None, uses config_multi_ue.SKLEARN_SEED
    stratify : bool, optional
        If True, maintain class proportions in both splits
        If None, uses config_multi_ue.STRATIFY_SPLIT
    verbose : bool, default=True
        If True, print split information

    Returns:
    --------
    X_train : np.ndarray
        Training features
    X_val : np.ndarray
        Validation features
    y_train : np.ndarray
        Training labels
    y_val : np.ndarray
        Validation labels

    Raises:
    -------
    ValueError
        If input arrays are empty or have mismatched lengths
    """

    # Set defaults from config
    if train_ratio is None:
        train_ratio = config_multi_ue.TRAIN_RATIO
    if random_seed is None:
        random_seed = config_multi_ue.SKLEARN_SEED
    if stratify is None:
        stratify = config_multi_ue.STRATIFY_SPLIT

    # Validate inputs
    if X.shape[0] == 0:
        raise ValueError("Input array X is empty")

    if y.shape[0] == 0:
        raise ValueError("Input array y is empty")

    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X and y have different number of samples: "
            f"X has {X.shape[0]}, y has {y.shape[0]}"
        )

    if X.shape[0] < 10:
        raise ValueError(
            f"Not enough samples for splitting: {X.shape[0]} samples. "
            f"Need at least 10."
        )

    # Validate train_ratio
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(
            f"train_ratio must be between 0 and 1, got {train_ratio}"
        )

    # Prepare stratify parameter
    stratify_array = y if stratify else None

    # Perform split
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        train_size=train_ratio,
        random_state=random_seed,
        stratify=stratify_array
    )

    # Print information
    if verbose:
        total = len(y)
        train_count = len(y_train)
        val_count = len(y_val)
        train_pct = train_count / total * 100
        val_pct = val_count / total * 100

        print(f"Split: {train_count:,} train, {val_count:,} validation "
              f"({train_pct:.1f}%/{val_pct:.1f}%)")

        # Show class distribution
        if stratify:
            print("Class distribution (stratified):")
            for name, y_set in [("  Train", y_train), ("  Val", y_val)]:
                unique, counts = np.unique(y_set, return_counts=True)
                dist = "  ".join(
                    [f"C{u}:{c:,}" for u, c in zip(unique, counts)])
                print(f"{name}: {dist}")

    return X_train, X_val, y_train, y_val


def prepare_data(
    datasets: Dict[int, Tuple[np.ndarray, np.ndarray]],
    train_snr: Optional[int] = None,
    train_ratio: Optional[float] = None,
    random_seed: Optional[int] = None,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """
    Prepare all data splits for training and testing.

    This function:
        1. Takes the training SNR dataset
        2. Splits it into training and validation sets
        3. Prepares test datasets for all SNR values

    Parameters:
    -----------
    datasets : dict
        Dictionary mapping SNR values to (X, y) tuples
        As returned by load_all_multi_ue_datasets()
    train_snr : int, optional
        SNR value to use for training
        If None, uses config_multi_ue.TRAIN_SNR
    train_ratio : float, optional
        Fraction of training SNR data for training
        If None, uses config_multi_ue.TRAIN_RATIO
    random_seed : int, optional
        Random seed for reproducibility
        If None, uses config_multi_ue.SKLEARN_SEED
    verbose : bool, default=True
        If True, print progress information

    Returns:
    --------
    X_train : np.ndarray
        Training features, shape (num_train, 24)
    y_train : np.ndarray
        Training labels, shape (num_train,)
    X_val : np.ndarray
        Validation features, shape (num_val, 24)
    y_val : np.ndarray
        Validation labels, shape (num_val,)
    X_test_dict : dict
        Dictionary mapping SNR to test features
        Key: int (SNR in dB)
        Value: np.ndarray of shape (num_samples, 24)
    y_test_dict : dict
        Dictionary mapping SNR to test labels
        Key: int (SNR in dB)
        Value: np.ndarray of shape (num_samples,)

    Raises:
    -------
    ValueError
        If train_snr is not in datasets
    """

    # Set defaults from config
    if train_snr is None:
        train_snr = config_multi_ue.TRAIN_SNR
    if train_ratio is None:
        train_ratio = config_multi_ue.TRAIN_RATIO
    if random_seed is None:
        random_seed = config_multi_ue.SKLEARN_SEED

    if verbose:
        print("\n" + "=" * 70)
        print("PREPARING DATA SPLITS (Multi-User)")
        print("=" * 70)

    # Validate train_snr exists in datasets
    if train_snr not in datasets:
        available_snrs = sorted(datasets.keys())
        raise ValueError(
            f"Training SNR {train_snr} dB not found in datasets. "
            f"Available SNR values: {available_snrs}"
        )

    # Get training SNR data
    X_full, y_full = datasets[train_snr]

    if verbose:
        print(f"\nTraining SNR: {train_snr} dB")
        print(f"Total samples: {len(X_full):,}")
        print(f"Scenario: {config_multi_ue.CURRENT_SCENARIO_KEY}")
        print(f"Number of users: {config_multi_ue.NUM_USERS}")

    # Split into training and validation
    X_train, X_val, y_train, y_val = split_train_validation(
        X=X_full,
        y=y_full,
        train_ratio=train_ratio,
        random_seed=random_seed,
        stratify=config_multi_ue.STRATIFY_SPLIT,
        verbose=verbose
    )

    # Prepare test datasets (all SNR values)
    X_test_dict = {}
    y_test_dict = {}

    if verbose:
        print(f"\nTest datasets:")

    for snr in sorted(datasets.keys()):
        X_test, y_test = datasets[snr]
        X_test_dict[snr] = X_test
        y_test_dict[snr] = y_test

        if verbose:
            print(f"  SNR {snr:2d} dB: {len(y_test):,} samples")

    if verbose:
        print("=" * 70 + "\n")

    return X_train, y_train, X_val, y_val, X_test_dict, y_test_dict


# =============================================================================
# SECTION 2: FEATURE NORMALIZATION
# =============================================================================

def normalize_features(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test_dict: Dict[int, np.ndarray],
    method: Optional[str] = None,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, Dict[int, np.ndarray], Any]:
    """
    Normalize features using specified method.

    IMPORTANT: The scaler is fitted ONLY on training data to prevent data leakage.
    The same transformation is then applied to validation and test data.

    Parameters:
    -----------
    X_train : np.ndarray
        Training features, shape (num_train, num_features)
    X_val : np.ndarray
        Validation features, shape (num_val, num_features)
    X_test_dict : dict
        Dictionary mapping SNR to test features
    method : str, optional
        Normalization method:
        - "standard": StandardScaler (zero mean, unit variance)
        - "minmax": MinMaxScaler (scale to 0-1 range)
        If None, uses config_multi_ue.NORMALIZATION_TYPE
    verbose : bool, default=True
        If True, print normalization information

    Returns:
    --------
    X_train_norm : np.ndarray
        Normalized training features, dtype float32
    X_val_norm : np.ndarray
        Normalized validation features, dtype float32
    X_test_dict_norm : dict
        Dictionary mapping SNR to normalized test features
    scaler : sklearn scaler object
        Fitted scaler (can be saved for later use)

    Raises:
    -------
    ValueError
        If method is not "standard" or "minmax"
    """

    # Set default method
    if method is None:
        method = config_multi_ue.NORMALIZATION_TYPE

    if verbose:
        print("\n--- Feature Normalization ---")

    # Create scaler based on method
    if method == "standard":
        scaler = StandardScaler()
        if verbose:
            print("Method: StandardScaler (zero mean, unit variance)")
    elif method == "minmax":
        scaler = MinMaxScaler(feature_range=(0, 1))
        if verbose:
            print("Method: MinMaxScaler (0-1 range)")
    else:
        raise ValueError(
            f"Unknown normalization method: '{method}'. "
            f"Use 'standard' or 'minmax'."
        )

    # Fit scaler on training data ONLY
    if verbose:
        print(f"Fitting scaler on training data ({len(X_train):,} samples)...")

    scaler.fit(X_train)

    # Transform training data
    X_train_norm = scaler.transform(X_train).astype(np.float32)

    # Transform validation data
    X_val_norm = scaler.transform(X_val).astype(np.float32)

    # Transform test data for each SNR
    X_test_dict_norm = {}
    for snr, X_test in X_test_dict.items():
        X_test_dict_norm[snr] = scaler.transform(X_test).astype(np.float32)

    # Print statistics
    if verbose:
        print(f"\nAfter normalization:")
        print(
            f"  Training   - Mean: {X_train_norm.mean():.6f}, Std: {X_train_norm.std():.6f}")
        print(
            f"  Validation - Mean: {X_val_norm.mean():.6f}, Std: {X_val_norm.std():.6f}")

    return X_train_norm, X_val_norm, X_test_dict_norm, scaler


def save_scaler(scaler: Any, filepath: Optional[str] = None,
                verbose: bool = True) -> str:
    """
    Save a fitted scaler to disk.

    Parameters:
    -----------
    scaler : sklearn scaler object
        Fitted scaler to save
    filepath : str, optional
        Path to save the scaler
        If None, uses config_multi_ue.SCALER_FILEPATH
    verbose : bool, default=True
        If True, print save confirmation

    Returns:
    --------
    filepath : str
        Path where scaler was saved
    """

    # Set default filepath
    if filepath is None:
        filepath = os.path.join(config_multi_ue.MODELS_DIR, config_multi_ue.SCALER_FILENAME)

    # Create directory if needed
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    # Save scaler
    with open(filepath, 'wb') as f:
        pickle.dump(scaler, f)

    if verbose:
        print(f"Scaler saved to: {filepath}")

    return filepath


def load_scaler(filepath: Optional[str] = None,
                verbose: bool = True) -> Any:
    """
    Load a saved scaler from disk.

    Parameters:
    -----------
    filepath : str, optional
        Path to the saved scaler
        If None, uses config_multi_ue.SCALER_FILEPATH
    verbose : bool, default=True
        If True, print load confirmation

    Returns:
    --------
    scaler : sklearn scaler object or None
        Loaded scaler, or None if file not found
    """

    # Set default filepath
    if filepath is None:
        filepath = config_multi_ue.SCALER_FILEPATH

    # Check if file exists
    if not os.path.exists(filepath):
        if verbose:
            print(f"Scaler file not found: {filepath}")
        return None

    # Load scaler
    with open(filepath, 'rb') as f:
        scaler = pickle.load(f)

    if verbose:
        print(f"Scaler loaded from: {filepath}")

    return scaler


# =============================================================================
# SECTION 3: COMPLETE PREPROCESSING PIPELINE
# =============================================================================

def preprocess_pipeline(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test_dict: Dict[int, np.ndarray],
    normalize: Optional[bool] = None,
    save_scaler_to_disk: bool = True,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, Dict[int, np.ndarray], Any]:
    """
    Complete preprocessing pipeline for Multi-User data.

    This function applies all preprocessing steps:
        1. Feature normalization (if enabled)
        2. Saves the scaler to disk (if normalization is used)

    Note: The paper does NOT mention normalization. By default, 
    config_multi_ue.NORMALIZE_FEATURES is set to False to match the paper.

    Parameters:
    -----------
    X_train : np.ndarray
        Training features
    X_val : np.ndarray
        Validation features
    X_test_dict : dict
        Dictionary mapping SNR to test features
    normalize : bool, optional
        Whether to normalize features
        If None, uses config_multi_ue.NORMALIZE_FEATURES
    save_scaler_to_disk : bool, default=True
        If True and normalization is used, save scaler to disk
    verbose : bool, default=True
        If True, print progress information

    Returns:
    --------
    X_train_proc : np.ndarray
        Processed training features
    X_val_proc : np.ndarray
        Processed validation features
    X_test_dict_proc : dict
        Processed test features for each SNR
    scaler : sklearn scaler object or None
        Fitted scaler if normalization was used, None otherwise
    """

    # Set default from config
    if normalize is None:
        normalize = config_multi_ue.NORMALIZE_FEATURES

    if verbose:
        print("\n" + "=" * 70)
        print("PREPROCESSING PIPELINE (Multi-User)")
        print("=" * 70)

    # Initialize scaler as None
    scaler = None

    # Apply normalization if enabled
    if normalize:
        if verbose:
            print("\nNormalization: ENABLED")

        X_train_proc, X_val_proc, X_test_dict_proc, scaler = normalize_features(
            X_train=X_train,
            X_val=X_val,
            X_test_dict=X_test_dict,
            method=config_multi_ue.NORMALIZATION_TYPE,
            verbose=verbose
        )

        # Save scaler to disk
        if save_scaler_to_disk:
            save_scaler(scaler, verbose=verbose)

    else:
        if verbose:
            print("\nNormalization: DISABLED (matching paper)")
            print("Note: The paper does not mention feature normalization.")
            print("Data will be used as-is from MATLAB.")

        # No preprocessing - return data as-is
        # Ensure float32 dtype for consistency
        X_train_proc = X_train.astype(np.float32)
        X_val_proc = X_val.astype(np.float32)
        X_test_dict_proc = {
            snr: X_test.astype(np.float32)
            for snr, X_test in X_test_dict.items()
        }

    if verbose:
        print("\nPreprocessing complete.")
        print("=" * 70 + "\n")

    return X_train_proc, X_val_proc, X_test_dict_proc, scaler


# =============================================================================
# SECTION 4: UTILITY FUNCTIONS
# =============================================================================

def get_data_statistics(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    verbose: bool = True
) -> Dict:
    """
    Get comprehensive statistics about the prepared data.

    Parameters:
    -----------
    X_train : np.ndarray
        Training features
    X_val : np.ndarray
        Validation features
    y_train : np.ndarray
        Training labels
    y_val : np.ndarray
        Validation labels
    verbose : bool, default=True
        If True, print statistics

    Returns:
    --------
    stats : dict
        Dictionary containing data statistics
    """

    stats = {}

    # Basic counts
    stats['num_train'] = int(len(y_train))
    stats['num_val'] = int(len(y_val))
    stats['num_features'] = int(X_train.shape[1])
    stats['train_ratio'] = float(len(y_train) / (len(y_train) + len(y_val)))

    # Class distributions
    train_unique, train_counts = np.unique(y_train, return_counts=True)
    val_unique, val_counts = np.unique(y_val, return_counts=True)

    stats['train_class_dist'] = {int(u): int(c)
                                 for u, c in zip(train_unique, train_counts)}
    stats['val_class_dist'] = {int(u): int(c)
                               for u, c in zip(val_unique, val_counts)}

    # Feature statistics
    stats['feature_stats'] = {
        'train_min': float(X_train.min()),
        'train_max': float(X_train.max()),
        'train_mean': float(X_train.mean()),
        'train_std': float(X_train.std()),
        'val_min': float(X_val.min()),
        'val_max': float(X_val.max()),
        'val_mean': float(X_val.mean()),
        'val_std': float(X_val.std())
    }

    if verbose:
        print("\n--- Data Statistics ---")
        print(f"Training samples: {stats['num_train']:,}")
        print(f"Validation samples: {stats['num_val']:,}")
        print(f"Features: {stats['num_features']}")
        print(f"Train ratio: {stats['train_ratio']:.2%}")

        print(f"\nTraining class distribution:")
        for c in sorted(stats['train_class_dist'].keys()):
            count = stats['train_class_dist'][c]
            pct = count / stats['num_train'] * 100
            print(f"  Class {c}: {count:,} ({pct:.2f}%)")

        print(f"\nFeature statistics:")
        fs = stats['feature_stats']
        print(f"  Train - Min: {fs['train_min']:.4f}, Max: {fs['train_max']:.4f}, "
              f"Mean: {fs['train_mean']:.4f}, Std: {fs['train_std']:.4f}")
        print(f"  Val   - Min: {fs['val_min']:.4f}, Max: {fs['val_max']:.4f}, "
              f"Mean: {fs['val_mean']:.4f}, Std: {fs['val_std']:.4f}")

    return stats


def verify_class_balance(y_train: np.ndarray, y_val: np.ndarray,
                         tolerance: float = 0.1) -> bool:
    """
    Verify that class distribution is balanced within tolerance.

    Parameters:
    -----------
    y_train : np.ndarray
        Training labels
    y_val : np.ndarray
        Validation labels
    tolerance : float, default=0.1
        Maximum allowed deviation from perfect balance (10%)

    Returns:
    --------
    is_balanced : bool
        True if both train and val sets are balanced within tolerance
    """

    # Check training set
    train_unique, train_counts = np.unique(y_train, return_counts=True)
    train_ratio = max(train_counts) / \
        min(train_counts) if min(train_counts) > 0 else float('inf')
    train_balanced = train_ratio < (1 + tolerance)

    # Check validation set
    val_unique, val_counts = np.unique(y_val, return_counts=True)
    val_ratio = max(val_counts) / \
        min(val_counts) if min(val_counts) > 0 else float('inf')
    val_balanced = val_ratio < (1 + tolerance)

    return train_balanced and val_balanced


# =============================================================================
# SECTION 5: SELF-TEST
# =============================================================================

if __name__ == "__main__":
    """
    Self-test for data_preprocessing_multi_ue module.
    """
    print("\n" + "=" * 70)
    print("DATA PREPROCESSING MULTI-UE MODULE - SELF TEST")
    print("=" * 70)

    # Create dummy data for testing
    print("\n--- Creating dummy data for testing ---")

    np.random.seed(config_multi_ue.MASTER_SEED)

    num_samples = 1000
    num_features = 24
    num_classes = 4

    # Create dummy features
    X_dummy = np.random.randn(num_samples, num_features).astype(np.float32)

    # Create dummy labels (balanced)
    y_dummy = np.repeat(np.arange(num_classes),
                        num_samples // num_classes).astype(np.int32)
    np.random.shuffle(y_dummy)

    print(f"Dummy data: X shape = {X_dummy.shape}, y shape = {y_dummy.shape}")
    print(
        f"Class distribution: {dict(zip(*np.unique(y_dummy, return_counts=True)))}")

    # Test 1: split_train_validation
    print("\n--- Test 1: split_train_validation ---")

    try:
        X_train, X_val, y_train, y_val = split_train_validation(
            X_dummy, y_dummy,
            train_ratio=0.75,
            random_seed=42,
            verbose=True
        )
        print(f"X_train: {X_train.shape}, X_val: {X_val.shape}")
        print(f"y_train: {y_train.shape}, y_val: {y_val.shape}")
        print("Test 1: PASS")
    except Exception as e:
        print(f"Test 1: FAIL - {e}")

    # Test 2: prepare_data
    print("\n--- Test 2: prepare_data ---")

    try:
        # Create dummy datasets dict
        dummy_datasets = {
            0: (X_dummy.copy(), y_dummy.copy()),
            5: (X_dummy.copy(), y_dummy.copy()),
            10: (X_dummy.copy(), y_dummy.copy()),
        }

        X_train, y_train, X_val, y_val, X_test_dict, y_test_dict = prepare_data(
            datasets=dummy_datasets,
            train_snr=10,
            verbose=True
        )

        print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
        print(f"Test SNRs: {list(X_test_dict.keys())}")
        print("Test 2: PASS")
    except Exception as e:
        print(f"Test 2: FAIL - {e}")

    # Test 3: normalize_features
    print("\n--- Test 3: normalize_features ---")

    try:
        X_train_norm, X_val_norm, X_test_dict_norm, scaler = normalize_features(
            X_train=X_train,
            X_val=X_val,
            X_test_dict=X_test_dict,
            method="standard",
            verbose=True
        )

        print(f"Normalized X_train mean: {X_train_norm.mean():.6f}")
        print(f"Normalized X_train std: {X_train_norm.std():.6f}")
        print(f"Scaler type: {type(scaler).__name__}")
        print("Test 3: PASS")
    except Exception as e:
        print(f"Test 3: FAIL - {e}")

    # Test 4: save_scaler and load_scaler
    print("\n--- Test 4: save_scaler and load_scaler ---")

    try:
        # Create test directory
        test_dir = "./test_temp_preproc/"
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)

        test_path = os.path.join(test_dir, "test_scaler.pkl")

        # Save
        save_scaler(scaler, filepath=test_path, verbose=True)

        # Load
        loaded_scaler = load_scaler(filepath=test_path, verbose=True)

        # Verify
        if loaded_scaler is not None:
            test_transform = loaded_scaler.transform(X_train[:5])
            original_transform = scaler.transform(X_train[:5])

            if np.allclose(test_transform, original_transform):
                print("Scaler save/load: MATCH")
                print("Test 4: PASS")
            else:
                print("Scaler save/load: MISMATCH")
                print("Test 4: FAIL")
        else:
            print("Test 4: FAIL - Could not load scaler")

        # Cleanup
        if os.path.exists(test_path):
            os.remove(test_path)
        if os.path.exists(test_dir):
            os.rmdir(test_dir)

    except Exception as e:
        print(f"Test 4: FAIL - {e}")

    # Test 5: preprocess_pipeline (without normalization)
    print("\n--- Test 5: preprocess_pipeline (no normalization) ---")

    try:
        X_train_proc, X_val_proc, X_test_dict_proc, scaler_none = preprocess_pipeline(
            X_train=X_train,
            X_val=X_val,
            X_test_dict=X_test_dict,
            normalize=False,
            verbose=True
        )

        if scaler_none is None:
            print("Scaler is None (correct)")

        if np.allclose(X_train, X_train_proc):
            print("Data unchanged (correct)")

        print("Test 5: PASS")
    except Exception as e:
        print(f"Test 5: FAIL - {e}")

    # Test 6: preprocess_pipeline (with normalization)
    print("\n--- Test 6: preprocess_pipeline (with normalization) ---")

    try:
        X_train_proc, X_val_proc, X_test_dict_proc, scaler_norm = preprocess_pipeline(
            X_train=X_train,
            X_val=X_val,
            X_test_dict=X_test_dict,
            normalize=True,
            save_scaler_to_disk=False,  # Don't save during test
            verbose=True
        )

        if scaler_norm is not None:
            print("Scaler is not None (correct)")

        print(f"Normalized mean: {X_train_proc.mean():.6f}")
        print("Test 6: PASS")
    except Exception as e:
        print(f"Test 6: FAIL - {e}")

    # Test 7: get_data_statistics
    print("\n--- Test 7: get_data_statistics ---")

    try:
        stats = get_data_statistics(
            X_train, X_val, y_train, y_val, verbose=True)
        print("Test 7: PASS")
    except Exception as e:
        print(f"Test 7: FAIL - {e}")

    # Test 8: verify_class_balance
    print("\n--- Test 8: verify_class_balance ---")

    try:
        is_balanced = verify_class_balance(y_train, y_val, tolerance=0.1)
        print(f"Is balanced: {is_balanced}")
        print("Test 8: PASS")
    except Exception as e:
        print(f"Test 8: FAIL - {e}")

    print("\n" + "=" * 70)
    print("SELF TEST COMPLETE")
    print("=" * 70)
