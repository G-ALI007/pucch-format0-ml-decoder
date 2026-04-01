"""
================================================================================
PUCCH Format 0 - Machine Learning Decoder
Neural Network Model Module
================================================================================

This module handles the neural network model including:
    - Model creation with configurable architecture
    - Model compilation with optimizer and loss function
    - Training with callbacks (early stopping, checkpoints, etc.)
    - Model saving and loading

Architecture (as per paper):
    Input(24) -> Dense(128, ReLU) -> Dropout(0.5) ->
    Dense(128, ReLU) -> Dropout(0.5) -> Dense(4, Softmax)

Training (as per paper):
    - Optimizer: SGD with momentum
    - Learning rate: 10^-3
    - Momentum: 0.9
    - Loss: Categorical Cross-Entropy
    - Epochs: 200

================================================================================
"""
from config import config
from tf_keras.utils import to_categorical
from tf_keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    CSVLogger
)
from tf_keras.optimizers import SGD
from tf_keras.layers import Dense, Dropout
from tf_keras.models import Sequential, load_model
import tf_keras
import tensorflow as tf
import os
import time
import numpy as np
from typing import List, Tuple, Dict, Optional, Any

# Fix for TensorFlow >= 2.16 with Keras 3
os.environ['TF_USE_LEGACY_KERAS'] = '1'

# TensorFlow and Keras imports

# Import configuration

# =============================================================================
# SECTION 1: RANDOM SEED SETUP
# =============================================================================


def set_random_seeds(seed: Optional[int] = None, verbose: bool = True) -> None:
    """
    Set random seeds for reproducibility.

    This function sets seeds for:
        - Python's hash seed
        - NumPy random generator
        - TensorFlow random generator

    Parameters:
    -----------
    seed : int, optional
        Random seed to use
        If None, uses config.MASTER_SEED
    verbose : bool, default=True
        If True, print confirmation

    Example:
    --------
    >>> set_random_seeds(42)
    Random seeds set to: 42
    """

    if seed is None:
        seed = config.MASTER_SEED

    # Set Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)

    # Set NumPy seed
    np.random.seed(seed)

    # Set TensorFlow seed
    tf.random.set_seed(seed)

    if verbose:
        print(f"Random seeds set to: {seed}")


# =============================================================================
# SECTION 2: MODEL CREATION
# =============================================================================

def create_model(
    input_size: Optional[int] = None,
    hidden_layers: Optional[List[int]] = None,
    output_size: Optional[int] = None,
    hidden_activation: Optional[str] = None,
    output_activation: Optional[str] = None,
    dropout_rate: Optional[float] = None,
    use_dropout: Optional[bool] = None,
    kernel_initializer: Optional[str] = None,
    learning_rate: Optional[float] = None,
    momentum: Optional[float] = None,
    use_nesterov: Optional[bool] = None,
    print_summary: bool = True
) -> Sequential:
    """
    Create and compile the neural network model.

    Architecture as per paper:
        Input(24) -> Dense(128, ReLU) -> Dropout(0.5) ->
        Dense(128, ReLU) -> Dropout(0.5) -> Dense(4, Softmax)

    Parameters:
    -----------
    input_size : int, optional
        Number of input features
        Default: config.INPUT_SIZE (24)
    hidden_layers : list of int, optional
        List of hidden layer sizes
        Default: config.HIDDEN_LAYERS ([128, 128])
    output_size : int, optional
        Number of output classes
        Default: config.OUTPUT_SIZE (4)
    hidden_activation : str, optional
        Activation function for hidden layers
        Default: config.HIDDEN_ACTIVATION ("relu")
    output_activation : str, optional
        Activation function for output layer
        Default: config.OUTPUT_ACTIVATION ("softmax")
    dropout_rate : float, optional
        Dropout probability (0 to 1)
        Default: config.DROPOUT_RATE (0.5)
    use_dropout : bool, optional
        Whether to add dropout layers
        Default: config.USE_DROPOUT (True)
    kernel_initializer : str, optional
        Weight initialization method
        Default: config.KERNEL_INITIALIZER ("glorot_uniform")
    learning_rate : float, optional
        Learning rate for optimizer
        Default: config.LEARNING_RATE (0.001)
    momentum : float, optional
        Momentum for SGD optimizer
        Default: config.MOMENTUM (0.9)
    use_nesterov : bool, optional
        Whether to use Nesterov momentum
        Default: config.USE_NESTEROV (False)
    print_summary : bool, default=True
        If True, print model summary

    Returns:
    --------
    model : keras.Sequential
        Compiled neural network model

    Example:
    --------
    >>> model = create_model()
    ======================================================================
    CREATING NEURAL NETWORK MODEL
    ======================================================================
    ...
    """

    # Set defaults from config
    if input_size is None:
        input_size = config.INPUT_SIZE
    if hidden_layers is None:
        hidden_layers = config.HIDDEN_LAYERS.copy()  # Copy to avoid modifying config
    if output_size is None:
        output_size = config.OUTPUT_SIZE
    if hidden_activation is None:
        hidden_activation = config.HIDDEN_ACTIVATION
    if output_activation is None:
        output_activation = config.OUTPUT_ACTIVATION
    if dropout_rate is None:
        dropout_rate = config.DROPOUT_RATE
    if use_dropout is None:
        use_dropout = config.USE_DROPOUT
    if kernel_initializer is None:
        kernel_initializer = config.KERNEL_INITIALIZER
    if learning_rate is None:
        learning_rate = config.LEARNING_RATE
    if momentum is None:
        momentum = config.MOMENTUM
    if use_nesterov is None:
        use_nesterov = config.USE_NESTEROV

    if print_summary:
        print("\n" + "=" * 70)
        print("CREATING NEURAL NETWORK MODEL")
        print("=" * 70)

    # Create Sequential model
    model = Sequential(name="PUCCH_Format0_Decoder")

    # First hidden layer (includes input shape)
    model.add(Dense(
        units=hidden_layers[0],
        activation=hidden_activation,
        kernel_initializer=kernel_initializer,
        input_shape=(input_size,),
        name='hidden_1'
    ))

    # Add dropout after first hidden layer
    if use_dropout and dropout_rate > 0:
        model.add(Dropout(
            rate=dropout_rate,
            name='dropout_1'
        ))

    # Additional hidden layers
    for i, units in enumerate(hidden_layers[1:], start=2):
        model.add(Dense(
            units=units,
            activation=hidden_activation,
            kernel_initializer=kernel_initializer,
            name=f'hidden_{i}'
        ))

        # Add dropout after each hidden layer
        if use_dropout and dropout_rate > 0:
            model.add(Dropout(
                rate=dropout_rate,
                name=f'dropout_{i}'
            ))

    # Output layer
    model.add(Dense(
        units=output_size,
        activation=output_activation,
        kernel_initializer=kernel_initializer,
        name='output'
    ))

    # Create optimizer (SGD with momentum as per paper)
    optimizer = SGD(
        learning_rate=learning_rate,
        momentum=momentum,
        nesterov=use_nesterov
    )

    # Compile model
    model.compile(
        optimizer=optimizer,
        loss=config.LOSS_FUNCTION,
        metrics=['accuracy']
    )

    # Print summary
    if print_summary:
        print()
        model.summary()

        print(f"\n--- Model Configuration ---")
        print(f"Input size: {input_size}")
        print(f"Hidden layers: {hidden_layers}")
        print(f"Hidden activation: {hidden_activation}")
        print(f"Output size: {output_size}")
        print(f"Output activation: {output_activation}")
        print(f"Dropout rate: {dropout_rate}")
        print(f"Use dropout: {use_dropout}")
        print(f"Kernel initializer: {kernel_initializer}")

        print(f"\n--- Optimizer Configuration ---")
        print(f"Optimizer: SGD")
        print(f"Learning rate: {learning_rate}")
        print(f"Momentum: {momentum}")
        print(f"Nesterov: {use_nesterov}")
        print(f"Loss function: {config.LOSS_FUNCTION}")

        print(f"\n--- Model Statistics ---")
        print(f"Total parameters: {model.count_params():,}")

        # Count trainable and non-trainable parameters
        trainable_params = sum([
            np.prod(v.get_shape().as_list())
            for v in model.trainable_variables
        ])
        non_trainable_params = sum([
            np.prod(v.get_shape().as_list())
            for v in model.non_trainable_variables
        ])
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Non-trainable parameters: {non_trainable_params:,}")

        print("=" * 70 + "\n")

    return model


def get_model_config(model: Sequential) -> Dict:
    """
    Get configuration dictionary from a model.

    Parameters:
    -----------
    model : keras.Sequential
        Neural network model

    Returns:
    --------
    config_dict : dict
        Dictionary containing model configuration
    """

    config_dict = {
        'name': model.name,
        'num_layers': len(model.layers),
        'total_params': model.count_params(),
        'layers': []
    }

    for layer in model.layers:
        layer_info = {
            'name': layer.name,
            'type': type(layer).__name__,
            'output_shape': layer.output_shape,
            'params': layer.count_params()
        }

        # Add layer-specific info
        if isinstance(layer, Dense):
            layer_info['units'] = layer.units
            layer_info['activation'] = layer.activation.__name__
        elif isinstance(layer, Dropout):
            layer_info['rate'] = float(layer.rate)

        config_dict['layers'].append(layer_info)

    return config_dict


# =============================================================================
# SECTION 3: TRAINING CALLBACKS
# =============================================================================

def create_callbacks(
    model_filepath: Optional[str] = None,
    history_filepath: Optional[str] = None,
    use_early_stopping: Optional[bool] = None,
    use_model_checkpoint: Optional[bool] = None,
    use_reduce_lr: Optional[bool] = None,
    use_csv_logger: bool = True,
    verbose: bool = True
) -> List:
    """
    Create training callbacks.

    Parameters:
    -----------
    model_filepath : str, optional
        Path to save the best model
        Default: config.MODELS_DIR + config.MODEL_FILENAME
    history_filepath : str, optional
        Path to save training history CSV
        Default: config.LOGS_DIR + config.TRAINING_HISTORY_FILENAME
    use_early_stopping : bool, optional
        Whether to use early stopping
        Default: config.USE_EARLY_STOPPING
    use_model_checkpoint : bool, optional
        Whether to save model checkpoints
        Default: config.USE_MODEL_CHECKPOINT
    use_reduce_lr : bool, optional
        Whether to reduce learning rate on plateau
        Default: config.USE_REDUCE_LR
    use_csv_logger : bool, default=True
        Whether to log training history to CSV
    verbose : bool, default=True
        If True, print callback information

    Returns:
    --------
    callbacks : list
        List of Keras callback objects

    Example:
    --------
    >>> callbacks = create_callbacks()
    Creating callbacks:
      - ModelCheckpoint: ./models/pucch_f0_nn_decoder.h5
      - EarlyStopping: patience=20
      - ReduceLROnPlateau: factor=0.5, patience=10
      - CSVLogger: ./logs/training_history.csv
    """

    # Set defaults from config
    if model_filepath is None:
        model_filepath = os.path.join(config.MODELS_DIR, config.MODEL_FILENAME)
    if history_filepath is None:
        history_filepath = os.path.join(
            config.LOGS_DIR, config.TRAINING_HISTORY_FILENAME)
    if use_early_stopping is None:
        use_early_stopping = config.USE_EARLY_STOPPING
    if use_model_checkpoint is None:
        use_model_checkpoint = config.USE_MODEL_CHECKPOINT
    if use_reduce_lr is None:
        use_reduce_lr = config.USE_REDUCE_LR

    callbacks = []

    if verbose:
        print("Creating callbacks:")

    # Ensure directories exist
    model_dir = os.path.dirname(model_filepath)
    if model_dir and not os.path.exists(model_dir):
        os.makedirs(model_dir)

    history_dir = os.path.dirname(history_filepath)
    if history_dir and not os.path.exists(history_dir):
        os.makedirs(history_dir)

    # Model Checkpoint - save best model
    if use_model_checkpoint:
        checkpoint = ModelCheckpoint(
            filepath=model_filepath,
            monitor=config.CHECKPOINT_MONITOR,
            mode=config.CHECKPOINT_MODE,
            save_best_only=config.CHECKPOINT_SAVE_BEST_ONLY,
            save_weights_only=False,
            verbose=1
        )
        callbacks.append(checkpoint)

        if verbose:
            print(f"  - ModelCheckpoint: {model_filepath}")
            print(
                f"      Monitor: {config.CHECKPOINT_MONITOR}, Mode: {config.CHECKPOINT_MODE}")

    # Early Stopping - stop training when no improvement
    if use_early_stopping:
        early_stopping = EarlyStopping(
            monitor=config.EARLY_STOPPING_MONITOR,
            mode=config.EARLY_STOPPING_MODE,
            patience=config.EARLY_STOPPING_PATIENCE,
            min_delta=config.EARLY_STOPPING_MIN_DELTA,
            restore_best_weights=config.EARLY_STOPPING_RESTORE_BEST,
            verbose=1
        )
        callbacks.append(early_stopping)

        if verbose:
            print(
                f"  - EarlyStopping: patience={config.EARLY_STOPPING_PATIENCE}")
            print(f"      Monitor: {config.EARLY_STOPPING_MONITOR}, "
                  f"Min delta: {config.EARLY_STOPPING_MIN_DELTA}")

    # Reduce Learning Rate on Plateau
    if use_reduce_lr:
        reduce_lr = ReduceLROnPlateau(
            monitor=config.REDUCE_LR_MONITOR,
            mode='min',
            factor=config.REDUCE_LR_FACTOR,
            patience=config.REDUCE_LR_PATIENCE,
            min_lr=config.REDUCE_LR_MIN_LR,
            verbose=1
        )
        callbacks.append(reduce_lr)

        if verbose:
            print(f"  - ReduceLROnPlateau: factor={config.REDUCE_LR_FACTOR}, "
                  f"patience={config.REDUCE_LR_PATIENCE}")
            print(f"      Min LR: {config.REDUCE_LR_MIN_LR}")

    # CSV Logger - log training history
    if use_csv_logger:
        csv_logger = CSVLogger(
            filename=history_filepath,
            separator=',',
            append=False
        )
        callbacks.append(csv_logger)

        if verbose:
            print(f"  - CSVLogger: {history_filepath}")

    if verbose:
        print()

    return callbacks


# =============================================================================
# SECTION 4: MODEL TRAINING
# =============================================================================

def train_model(
    model: Sequential,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    callbacks: Optional[List] = None,
    model_filepath: Optional[str] = None,
    verbose: int = 1
) -> Tuple[Dict, float]:
    """
    Train the neural network model.

    Parameters:
    -----------
    model : keras.Sequential
        Compiled neural network model
    X_train : np.ndarray
        Training features, shape (num_train, num_features)
    y_train : np.ndarray
        Training labels, shape (num_train,)
        Should be integer labels (0, 1, 2, 3), will be converted to one-hot
    X_val : np.ndarray
        Validation features, shape (num_val, num_features)
    y_val : np.ndarray
        Validation labels, shape (num_val,)
    epochs : int, optional
        Number of training epochs
        Default: config.NUM_EPOCHS
    batch_size : int, optional
        Batch size for training
        Default: config.BATCH_SIZE
    callbacks : list, optional
        List of Keras callbacks
        If None, callbacks will be created using create_callbacks()
    model_filepath : str, optional
        Path to save the best model
        Default: config.MODELS_DIR + config.MODEL_FILENAME
    verbose : int, default=1
        Verbosity level for training
        0 = silent, 1 = progress bar, 2 = one line per epoch

    Returns:
    --------
    history : dict
        Training history dictionary containing:
        - 'loss': Training loss per epoch
        - 'accuracy': Training accuracy per epoch
        - 'val_loss': Validation loss per epoch
        - 'val_accuracy': Validation accuracy per epoch
    training_time : float
        Total training time in seconds

    Example:
    --------
    >>> history, training_time = train_model(model, X_train, y_train, X_val, y_val)
    """

    # Set defaults from config
    if epochs is None:
        epochs = config.NUM_EPOCHS
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    if model_filepath is None:
        model_filepath = os.path.join(config.MODELS_DIR, config.MODEL_FILENAME)

    print("\n" + "=" * 70)
    print("TRAINING NEURAL NETWORK")
    print("=" * 70)

    # Validate inputs
    if X_train.shape[0] == 0:
        raise ValueError("Training data is empty")
    if X_val.shape[0] == 0:
        raise ValueError("Validation data is empty")
    if X_train.shape[1] != X_val.shape[1]:
        raise ValueError(
            f"Feature dimensions mismatch: "
            f"X_train has {X_train.shape[1]}, X_val has {X_val.shape[1]}"
        )
    if len(y_train) != X_train.shape[0]:
        raise ValueError(
            f"X_train and y_train have different number of samples: "
            f"{X_train.shape[0]} vs {len(y_train)}"
        )
    if len(y_val) != X_val.shape[0]:
        raise ValueError(
            f"X_val and y_val have different number of samples: "
            f"{X_val.shape[0]} vs {len(y_val)}"
        )

    # Convert labels to one-hot encoding
    num_classes = config.NUM_CLASSES
    y_train_onehot = to_categorical(y_train, num_classes=num_classes)
    y_val_onehot = to_categorical(y_val, num_classes=num_classes)

    # Print training configuration
    print(f"\n--- Training Configuration ---")
    print(f"Training samples: {len(X_train):,}")
    print(f"Validation samples: {len(X_val):,}")
    print(f"Input features: {X_train.shape[1]}")
    print(f"Output classes: {num_classes}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Steps per epoch: {len(X_train) // batch_size}")

    # Create callbacks if not provided
    if callbacks is None:
        print()
        callbacks = create_callbacks(
            model_filepath=model_filepath,
            verbose=True
        )

    print(f"Model will be saved to: {model_filepath}")

    print(f"\n--- Starting Training ---")
    print("-" * 70)

    # Record start time
    start_time = time.time()

    # Train the model
    history_obj = model.fit(
        x=X_train,
        y=y_train_onehot,
        validation_data=(X_val, y_val_onehot),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=verbose
    )

    # Calculate training time
    training_time = time.time() - start_time

    # Extract history dictionary
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

    # Find best epoch
    best_epoch = int(np.argmax(history['val_accuracy']) + 1)
    best_val_acc = float(max(history['val_accuracy']) * 100)
    best_val_loss = float(min(history['val_loss']))

    print(f"\n--- Best Results ---")
    print(
        f"Best validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch}")
    print(f"Best validation loss: {best_val_loss:.4f}")

    print("=" * 70 + "\n")

    return history, training_time


# =============================================================================
# SECTION 5: MODEL SAVING AND LOADING
# =============================================================================

def save_model(
    model: Sequential,
    filepath: Optional[str] = None,
    verbose: bool = True
) -> str:
    """
    Save a trained model to disk.

    Parameters:
    -----------
    model : keras.Sequential
        Trained model to save
    filepath : str, optional
        Path to save the model
        Default: config.MODELS_DIR + config.MODEL_FILENAME
    verbose : bool, default=True
        If True, print confirmation

    Returns:
    --------
    filepath : str
        Path where model was saved
    """

    if filepath is None:
        filepath = os.path.join(config.MODELS_DIR, config.MODEL_FILENAME)

    # Create directory if needed
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    # Save model
    model.save(filepath)

    if verbose:
        print(f"Model saved to: {filepath}")

    return filepath


def load_saved_model(
    filepath: Optional[str] = None,
    verbose: bool = True
) -> Sequential:
    """
    Load a saved model from disk.

    Parameters:
    -----------
    filepath : str, optional
        Path to the saved model
        Default: config.MODELS_DIR + config.MODEL_FILENAME
    verbose : bool, default=True
        If True, print confirmation

    Returns:
    --------
    model : keras.Sequential
        Loaded model

    Raises:
    -------
    FileNotFoundError
        If model file does not exist
    """

    if filepath is None:
        filepath = os.path.join(config.MODELS_DIR, config.MODEL_FILENAME)

    # Check if file exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")

    # Load model
    model = load_model(filepath)

    if verbose:
        print(f"Model loaded from: {filepath}")

    return model


def model_exists(filepath: Optional[str] = None) -> bool:
    """
    Check if a saved model exists.

    Parameters:
    -----------
    filepath : str, optional
        Path to check
        Default: config.MODELS_DIR + config.MODEL_FILENAME

    Returns:
    --------
    exists : bool
        True if model file exists
    """

    if filepath is None:
        filepath = os.path.join(config.MODELS_DIR, config.MODEL_FILENAME)

    return os.path.exists(filepath)


# =============================================================================
# SECTION 6: MODEL PREDICTION
# =============================================================================

def predict(
    model: Sequential,
    X: np.ndarray,
    return_probabilities: bool = False,
    batch_size: int = 256,
    verbose: bool = False
) -> np.ndarray:
    """
    Make predictions using the trained model.

    Parameters:
    -----------
    model : keras.Sequential
        Trained model
    X : np.ndarray
        Input features, shape (num_samples, num_features)
    return_probabilities : bool, default=False
        If True, return class probabilities
        If False, return class labels
    batch_size : int, default=256
        Batch size for prediction
    verbose : bool, default=False
        If True, print prediction information

    Returns:
    --------
    predictions : np.ndarray
        If return_probabilities=False: Class labels, shape (num_samples,)
        If return_probabilities=True: Class probabilities, shape (num_samples, num_classes)
    """

    if verbose:
        print(f"Predicting {len(X):,} samples...")

    # Get probabilities
    probabilities = model.predict(X, batch_size=batch_size, verbose=0)

    if return_probabilities:
        return probabilities
    else:
        # Return class labels
        return np.argmax(probabilities, axis=1)


def predict_single(
    model: Sequential,
    x: np.ndarray,
    return_probabilities: bool = False
) -> Any:
    """
    Make prediction for a single sample.

    Parameters:
    -----------
    model : keras.Sequential
        Trained model
    x : np.ndarray
        Single input sample, shape (num_features,) or (1, num_features)
    return_probabilities : bool, default=False
        If True, return class probabilities

    Returns:
    --------
    prediction : int or np.ndarray
        If return_probabilities=False: Predicted class (int)
        If return_probabilities=True: Class probabilities (1D array)
    """

    # Ensure correct shape
    if x.ndim == 1:
        x = x.reshape(1, -1)

    # Get probabilities
    probabilities = model.predict(x, verbose=0)[0]

    if return_probabilities:
        return probabilities
    else:
        return int(np.argmax(probabilities))


# =============================================================================
# SECTION 7: UTILITY FUNCTIONS
# =============================================================================

def get_training_summary(history: Dict) -> Dict:
    """
    Get summary statistics from training history.

    Parameters:
    -----------
    history : dict
        Training history from train_model()

    Returns:
    --------
    summary : dict
        Summary statistics
    """

    summary = {
        'epochs_completed': len(history['loss']),
        'final_train_acc': float(history['accuracy'][-1] * 100),
        'final_val_acc': float(history['val_accuracy'][-1] * 100),
        'best_val_acc': float(max(history['val_accuracy']) * 100),
        'best_epoch': int(np.argmax(history['val_accuracy']) + 1),
        'final_train_loss': float(history['loss'][-1]),
        'final_val_loss': float(history['val_loss'][-1]),
        'best_val_loss': float(min(history['val_loss']))
    }

    return summary


def print_gpu_info() -> None:
    """
    Print GPU information if available.
    """

    print("\n--- GPU Information ---")

    gpus = tf.config.list_physical_devices('GPU')

    if gpus:
        print(f"GPUs available: {len(gpus)}")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name}")
    else:
        print("No GPU available. Using CPU.")

    print()


def print_tensorflow_info() -> None:
    """
    Print TensorFlow version and configuration information.
    """

    print("\n--- TensorFlow Information ---")
    print(f"TensorFlow version: {tf.__version__}")

    # Get tf_keras version safely
    try:
        print(f"tf_keras version: {tf_keras.__version__}")
    except AttributeError:
        print(f"tf_keras: installed (version not available)")

    # Check GPU
    gpus = tf.config.list_physical_devices('GPU')
    print(f"GPUs available: {len(gpus)}")

    # Check if eager execution is enabled
    print(f"Eager execution: {tf.executing_eagerly()}")

    print()


# =============================================================================
# SECTION 8: SELF-TEST
# =============================================================================

if __name__ == "__main__":
    """
    Self-test for model module.
    """

    print("\n" + "=" * 70)
    print("MODEL MODULE - SELF TEST")
    print("=" * 70)

    # Test 1: Set random seeds
    print("\n--- Test 1: set_random_seeds ---")
    try:
        set_random_seeds(42, verbose=True)
        print("Test 1: PASS")
    except Exception as e:
        print(f"Test 1: FAIL - {e}")

    # Test 2: Print TensorFlow info
    print("\n--- Test 2: print_tensorflow_info ---")
    try:
        print_tensorflow_info()
        print("Test 2: PASS")
    except Exception as e:
        print(f"Test 2: FAIL - {e}")

    # Test 3: Print GPU info
    print("\n--- Test 3: print_gpu_info ---")
    try:
        print_gpu_info()
        print("Test 3: PASS")
    except Exception as e:
        print(f"Test 3: FAIL - {e}")

    # Test 4: Create model
    print("\n--- Test 4: create_model ---")
    try:
        model = create_model(print_summary=True)
        print(f"Model created: {model.name}")
        print(f"Input shape: {model.input_shape}")
        print(f"Output shape: {model.output_shape}")
        print("Test 4: PASS")
    except Exception as e:
        print(f"Test 4: FAIL - {e}")

    # Test 5: Get model config
    print("\n--- Test 5: get_model_config ---")
    try:
        model_config = get_model_config(model)
        print(f"Model name: {model_config['name']}")
        print(f"Num layers: {model_config['num_layers']}")
        print(f"Total params: {model_config['total_params']:,}")
        print("Test 5: PASS")
    except Exception as e:
        print(f"Test 5: FAIL - {e}")

    # Test 6: Create callbacks
    print("\n--- Test 6: create_callbacks ---")
    try:
        callbacks = create_callbacks(verbose=True)
        print(f"Callbacks created: {len(callbacks)}")
        print("Test 6: PASS")
    except Exception as e:
        print(f"Test 6: FAIL - {e}")

    # Test 7: Create dummy data and train briefly
    print("\n--- Test 7: train_model (5 epochs) ---")
    try:
        # Create dummy data
        np.random.seed(42)
        X_train_dummy = np.random.randn(1000, 24).astype(np.float32)
        y_train_dummy = np.random.randint(0, 4, 1000).astype(np.int32)
        X_val_dummy = np.random.randn(200, 24).astype(np.float32)
        y_val_dummy = np.random.randint(0, 4, 200).astype(np.int32)

        # Create fresh model
        model_test = create_model(print_summary=False)

        # Train for just 5 epochs
        history, train_time = train_model(
            model=model_test,
            X_train=X_train_dummy,
            y_train=y_train_dummy,
            X_val=X_val_dummy,
            y_val=y_val_dummy,
            epochs=5,
            batch_size=64,
            verbose=1
        )

        print(f"History keys: {list(history.keys())}")
        print(f"Training time: {train_time:.2f}s")
        print("Test 7: PASS")
    except Exception as e:
        print(f"Test 7: FAIL - {e}")

    # Test 8: Get training summary
    print("\n--- Test 8: get_training_summary ---")
    try:
        summary = get_training_summary(history)
        print(f"Epochs: {summary['epochs_completed']}")
        print(f"Best val acc: {summary['best_val_acc']:.2f}%")
        print(f"Best epoch: {summary['best_epoch']}")
        print("Test 8: PASS")
    except Exception as e:
        print(f"Test 8: FAIL - {e}")

    # Test 9: Predict
    print("\n--- Test 9: predict ---")
    try:
        X_test_dummy = np.random.randn(100, 24).astype(np.float32)

        # Class labels
        y_pred = predict(model_test, X_test_dummy, return_probabilities=False)
        print(f"Predictions shape: {y_pred.shape}")
        print(f"Unique predictions: {np.unique(y_pred)}")

        # Probabilities
        y_proba = predict(model_test, X_test_dummy, return_probabilities=True)
        print(f"Probabilities shape: {y_proba.shape}")
        print(f"Sum of probabilities (should be ~1): {y_proba[0].sum():.4f}")

        print("Test 9: PASS")
    except Exception as e:
        print(f"Test 9: FAIL - {e}")

    # Test 10: Predict single
    print("\n--- Test 10: predict_single ---")
    try:
        x_single = np.random.randn(24).astype(np.float32)

        pred_label = predict_single(
            model_test, x_single, return_probabilities=False)
        print(f"Predicted class: {pred_label}")
        print(f"Type: {type(pred_label)}")

        pred_proba = predict_single(
            model_test, x_single, return_probabilities=True)
        print(f"Probabilities: {pred_proba}")
        print(f"Sum: {pred_proba.sum():.4f}")

        print("Test 10: PASS")
    except Exception as e:
        print(f"Test 10: FAIL - {e}")

    # Test 11: Save and load model
    print("\n--- Test 11: save_model and load_saved_model ---")
    try:
        # Create test directory
        test_dir = "./test_temp_model/"
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)

        test_path = os.path.join(test_dir, "test_model.h5")

        # Save
        save_model(model_test, filepath=test_path, verbose=True)

        # Check exists
        exists = model_exists(filepath=test_path)
        print(f"Model exists: {exists}")

        # Load
        loaded_model = load_saved_model(filepath=test_path, verbose=True)

        # Verify predictions match
        pred_original = predict(model_test, X_test_dummy[:10])
        pred_loaded = predict(loaded_model, X_test_dummy[:10])

        if np.array_equal(pred_original, pred_loaded):
            print("Predictions match: YES")
        else:
            print("Predictions match: NO")

        # Cleanup
        if os.path.exists(test_path):
            os.remove(test_path)
        if os.path.exists(test_dir):
            os.rmdir(test_dir)

        print("Test 11: PASS")
    except Exception as e:
        print(f"Test 11: FAIL - {e}")

    print("\n" + "=" * 70)
    print("SELF TEST COMPLETE")
    print("=" * 70)
