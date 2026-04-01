"""
================================================================================
PUCCH Format 0 - Machine Learning Decoder
Visualization Module
================================================================================

This module handles all plotting and visualization including:
    - Training history plots (accuracy and loss curves)
    - Accuracy vs SNR plots
    - Confusion matrices heatmaps
    - Comparison plots (NN vs Correlation decoder)
    - Per-class accuracy plots

All plots can be displayed and/or saved to disk.

================================================================================
"""

import os
import numpy as np
from typing import Dict, Optional

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Import configuration
from config import config


# =============================================================================
# SECTION 1: PLOT STYLE SETUP
# =============================================================================

def setup_plot_style():
    """
    Set up consistent plot style for all figures.
    This function should be called once at the beginning.
    """

    plt.style.use('default')

    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 11,
        'figure.titlesize': 16,
        'figure.dpi': 100,
        'savefig.dpi': config.FIGURE_DPI,
        'savefig.bbox': 'tight'
    })


# Initialize plot style
setup_plot_style()


# =============================================================================
# SECTION 2: HELPER FUNCTIONS
# =============================================================================

def _save_and_show(fig, save_path: Optional[str], show_plot: bool):
    """
    Helper function to save and/or show a figure.

    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        Figure to save/show
    save_path : str or None
        Path to save the figure. If None, figure is not saved.
    show_plot : bool
        If True, display the figure
    """

    if save_path is not None:
        # Create directory if needed
        directory = os.path.dirname(save_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        fig.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def _get_default_save_path(filename: str) -> str:
    """
    Get default save path for a plot.

    Parameters:
    -----------
    filename : str
        Name of the file (e.g., "training_history.png")

    Returns:
    --------
    filepath : str
        Full path in plots directory
    """

    return os.path.join(config.PLOTS_DIR, filename)


# =============================================================================
# SECTION 3: TRAINING HISTORY PLOTS
# =============================================================================

def plot_training_history(
    history: Dict,
    save_path: Optional[str] = None,
    show_plot: bool = True
):
    """
    Plot training and validation accuracy and loss curves.

    Creates a figure with two subplots:
        - Left: Training and validation accuracy vs epoch
        - Right: Training and validation loss vs epoch

    Parameters:
    -----------
    history : dict
        Training history dictionary containing:
        - 'accuracy': Training accuracy per epoch
        - 'val_accuracy': Validation accuracy per epoch
        - 'loss': Training loss per epoch
        - 'val_loss': Validation loss per epoch
    save_path : str, optional
        Path to save the figure
        If None, uses default path in plots directory
    show_plot : bool, default=True
        If True, display the figure

    Example:
    --------
    >>> plot_training_history(history)
    """

    # Set default save path
    if save_path is None:
        save_path = _get_default_save_path("training_history.png")

    # Validate history keys
    required_keys = ['accuracy', 'val_accuracy', 'loss', 'val_loss']
    for key in required_keys:
        if key not in history:
            raise ValueError(f"Missing key '{key}' in history dictionary")

    # Prepare data
    epochs = range(1, len(history['accuracy']) + 1)
    train_acc = [a * 100 for a in history['accuracy']]
    val_acc = [a * 100 for a in history['val_accuracy']]
    train_loss = history['loss']
    val_loss = history['val_loss']

    # Find best epoch
    best_epoch = int(np.argmax(history['val_accuracy']) + 1)
    best_val_acc = float(max(val_acc))

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=config.FIGURE_SIZE_LARGE)

    # --- Left plot: Accuracy ---
    axes[0].plot(epochs, train_acc, 'b-', linewidth=2, label='Training')
    axes[0].plot(epochs, val_acc, 'r-', linewidth=2, label='Validation')

    # Mark best epoch
    axes[0].axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7,
                    label=f'Best epoch ({best_epoch})')
    axes[0].scatter([best_epoch], [best_val_acc], color='green', s=100,
                    zorder=5, edgecolors='black', linewidths=1)
    axes[0].annotate(
        f'{best_val_acc:.2f}%',
        xy=(best_epoch, best_val_acc),
        xytext=(10, -15),
        textcoords='offset points',
        fontsize=10,
        fontweight='bold',
        color='green'
    )

    axes[0].set_xlabel('Epoch', fontsize=config.FONT_SIZE_MEDIUM)
    axes[0].set_ylabel('Accuracy (%)', fontsize=config.FONT_SIZE_MEDIUM)
    axes[0].set_title('Model Accuracy', fontsize=config.FONT_SIZE_LARGE)
    axes[0].legend(fontsize=config.FONT_SIZE_SMALL)
    axes[0].grid(True, alpha=0.3)

    # --- Right plot: Loss ---
    axes[1].plot(epochs, train_loss, 'b-', linewidth=2, label='Training')
    axes[1].plot(epochs, val_loss, 'r-', linewidth=2, label='Validation')

    axes[1].set_xlabel('Epoch', fontsize=config.FONT_SIZE_MEDIUM)
    axes[1].set_ylabel('Categorical Cross-Entropy Loss',
                       fontsize=config.FONT_SIZE_MEDIUM)
    axes[1].set_title('Model Loss', fontsize=config.FONT_SIZE_LARGE)
    axes[1].legend(fontsize=config.FONT_SIZE_SMALL)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    _save_and_show(fig, save_path, show_plot)


def plot_accuracy_only(
    history: Dict,
    save_path: Optional[str] = None,
    show_plot: bool = True
):
    """
    Plot only the accuracy curve (single plot, matching paper's Figure 4a).

    Parameters:
    -----------
    history : dict
        Training history dictionary
    save_path : str, optional
        Path to save figure
    show_plot : bool, default=True
        If True, display the figure
    """

    if save_path is None:
        save_path = _get_default_save_path("accuracy_curve.png")

    epochs = range(1, len(history['accuracy']) + 1)
    train_acc = [a * 100 for a in history['accuracy']]
    val_acc = [a * 100 for a in history['val_accuracy']]

    fig, ax = plt.subplots(figsize=config.FIGURE_SIZE_MEDIUM)

    ax.plot(epochs, train_acc, 'b-o', linewidth=2,
            markersize=3, label='Training')
    ax.plot(epochs, val_acc, 'r-s', linewidth=2,
            markersize=3, label='Validation')

    ax.set_xlabel('Training Epoch', fontsize=config.FONT_SIZE_MEDIUM)
    ax.set_ylabel('Accuracy (%)', fontsize=config.FONT_SIZE_MEDIUM)
    ax.set_title(f'Model Accuracy, SNR = {config.TRAIN_SNR} dB',
                 fontsize=config.FONT_SIZE_LARGE)
    ax.legend(fontsize=config.FONT_SIZE_SMALL)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    _save_and_show(fig, save_path, show_plot)


def plot_loss_only(
    history: Dict,
    save_path: Optional[str] = None,
    show_plot: bool = True
):
    """
    Plot only the loss curve (single plot, matching paper's Figure 4b).

    Parameters:
    -----------
    history : dict
        Training history dictionary
    save_path : str, optional
        Path to save figure
    show_plot : bool, default=True
        If True, display the figure
    """

    if save_path is None:
        save_path = _get_default_save_path("loss_curve.png")

    epochs = range(1, len(history['loss']) + 1)

    fig, ax = plt.subplots(figsize=config.FIGURE_SIZE_MEDIUM)

    ax.plot(epochs, history['loss'], 'b-o',
            linewidth=2, markersize=3, label='Training')
    ax.plot(epochs, history['val_loss'], 'r-s',
            linewidth=2, markersize=3, label='Validation')

    ax.set_xlabel('Training Epoch', fontsize=config.FONT_SIZE_MEDIUM)
    ax.set_ylabel('Categorical Cross Entropy Loss',
                  fontsize=config.FONT_SIZE_MEDIUM)
    ax.set_title(f'Model Loss, SNR = {config.TRAIN_SNR} dB',
                 fontsize=config.FONT_SIZE_LARGE)
    ax.legend(fontsize=config.FONT_SIZE_SMALL)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    _save_and_show(fig, save_path, show_plot)


# =============================================================================
# SECTION 4: ACCURACY VS SNR PLOTS
# =============================================================================

def plot_accuracy_vs_snr(
    all_results: Dict[int, Dict],
    comparison: Optional[Dict] = None,
    save_path: Optional[str] = None,
    show_plot: bool = True
):
    """
    Plot accuracy vs SNR curve (similar to paper's Figure 5).

    If comparison data is provided, both NN and correlation decoder
    accuracies are plotted on the same figure.

    Parameters:
    -----------
    all_results : dict
        Neural network results for each SNR
        As returned by evaluate_all_snr()
    comparison : dict, optional
        Comparison results from compare_nn_vs_correlation()
        If provided, correlation decoder results are also plotted
    save_path : str, optional
        Path to save the figure
    show_plot : bool, default=True
        If True, display the figure

    Example:
    --------
    >>> plot_accuracy_vs_snr(all_results, comparison)
    """

    if save_path is None:
        save_path = _get_default_save_path("accuracy_vs_snr.png")

    # Extract data
    snr_values = sorted(all_results.keys())
    nn_accuracies = [all_results[snr]['accuracy'] * 100 for snr in snr_values]

    # Create figure
    fig, ax = plt.subplots(figsize=config.FIGURE_SIZE_MEDIUM)

    # Plot NN results
    ax.plot(snr_values, nn_accuracies, 'b-o', linewidth=2.5, markersize=10,
            label='Neural Network', markerfacecolor='white', markeredgewidth=2)

    # Plot correlation results if available
    if comparison is not None:
        corr_accuracies = comparison['corr_accuracy']
        ax.plot(snr_values, corr_accuracies, 'r--s', linewidth=2.5, markersize=10,
                label='Correlation Decoder', markerfacecolor='white', markeredgewidth=2)

    # Add 3GPP requirement line
    ax.axhline(y=99, color='green', linestyle=':', linewidth=2,
               label='3GPP Requirement (99%)', alpha=0.8)

    # Add value labels on NN points
    for snr, acc in zip(snr_values, nn_accuracies):
        ax.annotate(
            f'{acc:.1f}%',
            xy=(snr, acc),
            xytext=(0, 12),
            textcoords='offset points',
            ha='center',
            fontsize=9,
            fontweight='bold',
            color='blue'
        )

    # Configure axes
    ax.set_xlabel('SNR (dB)', fontsize=config.FONT_SIZE_LARGE)
    ax.set_ylabel('Accuracy (%)', fontsize=config.FONT_SIZE_LARGE)
    ax.set_title('Model Accuracy', fontsize=config.FONT_SIZE_TITLE)
    ax.legend(fontsize=config.FONT_SIZE_SMALL, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(snr_values)

    # Set y-axis limits
    min_acc = min(nn_accuracies)
    if comparison is not None:
        min_acc = min(min_acc, min(corr_accuracies))
    ax.set_ylim([max(0, min_acc - 5), 102])

    plt.tight_layout()

    _save_and_show(fig, save_path, show_plot)


# =============================================================================
# SECTION 5: CONFUSION MATRIX PLOTS
# =============================================================================

def plot_confusion_matrices_all(
    all_results: Dict[int, Dict],
    save_path: Optional[str] = None,
    show_plot: bool = True
):
    """
    Plot confusion matrices for all SNR values in a single figure.

    Creates a row of heatmaps, one for each SNR value.

    Parameters:
    -----------
    all_results : dict
        Results for each SNR from evaluate_all_snr()
    save_path : str, optional
        Path to save the figure
    show_plot : bool, default=True
        If True, display the figure

    Example:
    --------
    >>> plot_confusion_matrices_all(all_results)
    """

    if save_path is None:
        save_path = _get_default_save_path("confusion_matrices_all.png")

    snr_values = sorted(all_results.keys())
    num_snr = len(snr_values)

    # Create figure
    fig_width = 4.5 * num_snr
    fig_height = 4
    fig, axes = plt.subplots(1, num_snr, figsize=(fig_width, fig_height))

    # Handle single SNR case
    if num_snr == 1:
        axes = [axes]

    # Class labels for axes
    class_labels = [f'C{i}' for i in range(config.NUM_CLASSES)]

    for idx, snr in enumerate(snr_values):
        conf_matrix = all_results[snr]['confusion_matrix']
        accuracy = all_results[snr]['accuracy'] * 100

        # Plot heatmap
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_labels,
            yticklabels=class_labels,
            ax=axes[idx],
            cbar=True,
            annot_kws={"size": 10},
            linewidths=0.5,
            linecolor='gray'
        )

        axes[idx].set_xlabel('Predicted', fontsize=config.FONT_SIZE_SMALL)
        axes[idx].set_ylabel('Actual', fontsize=config.FONT_SIZE_SMALL)
        axes[idx].set_title(
            f'SNR = {snr} dB\nAcc = {accuracy:.1f}%',
            fontsize=config.FONT_SIZE_MEDIUM
        )

    plt.tight_layout()

    _save_and_show(fig, save_path, show_plot)


def plot_confusion_matrix_single(
    conf_matrix: np.ndarray,
    snr_db: int,
    accuracy: float,
    save_path: Optional[str] = None,
    show_plot: bool = True
):
    """
    Plot a single confusion matrix with detailed annotations.

    Parameters:
    -----------
    conf_matrix : np.ndarray
        Confusion matrix, shape (num_classes, num_classes)
    snr_db : int
        SNR value in dB
    accuracy : float
        Overall accuracy (0 to 1)
    save_path : str, optional
        Path to save the figure
    show_plot : bool, default=True
        If True, display the figure

    Example:
    --------
    >>> plot_confusion_matrix_single(
    ...     conf_matrix=results['confusion_matrix'],
    ...     snr_db=10,
    ...     accuracy=results['accuracy']
    ... )
    """

    if save_path is None:
        save_path = _get_default_save_path(f"confusion_matrix_{snr_db}dB.png")

    # Create figure
    fig, ax = plt.subplots(figsize=config.FIGURE_SIZE_SMALL)

    # Class labels
    class_labels = [f'Class {i}' for i in range(config.NUM_CLASSES)]

    # Plot heatmap
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_labels,
        yticklabels=class_labels,
        ax=ax,
        cbar=True,
        annot_kws={"size": 12, "fontweight": "bold"},
        linewidths=1,
        linecolor='gray'
    )

    ax.set_xlabel('Predicted Class', fontsize=config.FONT_SIZE_MEDIUM)
    ax.set_ylabel('Actual Class', fontsize=config.FONT_SIZE_MEDIUM)
    ax.set_title(
        f'Confusion Matrix - SNR = {snr_db} dB (Accuracy = {accuracy*100:.2f}%)',
        fontsize=config.FONT_SIZE_LARGE
    )

    plt.tight_layout()

    _save_and_show(fig, save_path, show_plot)


def plot_selected_confusion_matrices(
    all_results: Dict[int, Dict],
    selected_snrs: Optional[list] = None,
    save_path: Optional[str] = None,
    show_plot: bool = True
):
    """
    Plot confusion matrices for selected SNR values.

    Parameters:
    -----------
    all_results : dict
        Results from evaluate_all_snr()
    selected_snrs : list, optional
        List of SNR values to plot
        Default: first, middle, and last SNR values
    save_path : str, optional
        Path to save the figure
    show_plot : bool, default=True
        If True, display the figure
    """

    if save_path is None:
        save_path = _get_default_save_path("confusion_matrices_selected.png")

    # Select SNR values if not specified
    if selected_snrs is None:
        snr_values = sorted(all_results.keys())
        if len(snr_values) >= 3:
            selected_snrs = [snr_values[0],
                             snr_values[len(snr_values)//2], snr_values[-1]]
        else:
            selected_snrs = snr_values

    # Filter to available SNRs
    selected_snrs = [snr for snr in selected_snrs if snr in all_results]
    num_plots = len(selected_snrs)

    if num_plots == 0:
        print("Warning: No valid SNR values selected for confusion matrix plot")
        return

    # Create figure
    fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 4.5))

    if num_plots == 1:
        axes = [axes]

    class_labels = [f'C{i}' for i in range(config.NUM_CLASSES)]

    for idx, snr in enumerate(selected_snrs):
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
            annot_kws={"size": 11},
            linewidths=0.5,
            linecolor='gray'
        )

        axes[idx].set_xlabel('Predicted', fontsize=config.FONT_SIZE_SMALL)
        axes[idx].set_ylabel('Actual', fontsize=config.FONT_SIZE_SMALL)
        axes[idx].set_title(
            f'SNR = {snr} dB\nAcc = {accuracy:.1f}%',
            fontsize=config.FONT_SIZE_MEDIUM
        )

    plt.tight_layout()

    _save_and_show(fig, save_path, show_plot)


# =============================================================================
# SECTION 6: COMPARISON PLOTS
# =============================================================================

def plot_comparison(
    comparison: Dict,
    save_path: Optional[str] = None,
    show_plot: bool = True
):
    """
    Plot comparison between NN and correlation decoder.

    Creates a figure with two subplots:
        - Left: Accuracy comparison
        - Right: Accuracy gain (NN - Correlation)

    Parameters:
    -----------
    comparison : dict
        Comparison results from compare_nn_vs_correlation()
    save_path : str, optional
        Path to save the figure
    show_plot : bool, default=True
        If True, display the figure

    Example:
    --------
    >>> plot_comparison(comparison)
    """

    if save_path is None:
        save_path = _get_default_save_path("nn_vs_correlation.png")

    snr_values = comparison['snr']
    nn_acc = comparison['nn_accuracy']
    corr_acc = comparison['corr_accuracy']
    gains = [n - c for n, c in zip(nn_acc, corr_acc)]

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=config.FIGURE_SIZE_LARGE)

    # --- Left: Accuracy comparison ---
    axes[0].plot(snr_values, nn_acc, 'b-o', linewidth=2.5, markersize=10,
                 label='Neural Network', markerfacecolor='white', markeredgewidth=2)
    axes[0].plot(snr_values, corr_acc, 'r--s', linewidth=2.5, markersize=10,
                 label='Correlation Decoder', markerfacecolor='white', markeredgewidth=2)

    # 3GPP line
    axes[0].axhline(y=99, color='green', linestyle=':', linewidth=2,
                    label='3GPP Requirement (99%)', alpha=0.8)

    axes[0].set_xlabel('SNR (dB)', fontsize=config.FONT_SIZE_LARGE)
    axes[0].set_ylabel('Accuracy (%)', fontsize=config.FONT_SIZE_LARGE)
    axes[0].set_title('Accuracy Comparison', fontsize=config.FONT_SIZE_TITLE)
    axes[0].legend(fontsize=config.FONT_SIZE_SMALL, loc='lower right')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(snr_values)

    # Set y-axis limits
    min_acc = min(min(nn_acc), min(corr_acc))
    axes[0].set_ylim([max(0, min_acc - 5), 102])

    # --- Right: Gain ---
    colors = ['green' if g >= 0 else 'red' for g in gains]
    bars = axes[1].bar(snr_values, gains, color=colors, edgecolor='black',
                       linewidth=1.5, width=3)

    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=1)

    # Add value labels on bars
    for snr, gain, bar in zip(snr_values, gains, bars):
        va = 'bottom' if gain >= 0 else 'top'
        offset = 0.3 if gain >= 0 else -0.3
        axes[1].annotate(
            f'{gain:+.2f}%',
            xy=(snr, gain + offset),
            ha='center',
            va=va,
            fontsize=10,
            fontweight='bold'
        )

    axes[1].set_xlabel('SNR (dB)', fontsize=config.FONT_SIZE_LARGE)
    axes[1].set_ylabel('Accuracy Gain (%)', fontsize=config.FONT_SIZE_LARGE)
    axes[1].set_title('NN Gain over Correlation Decoder',
                      fontsize=config.FONT_SIZE_TITLE)
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].set_xticks(snr_values)

    plt.tight_layout()

    _save_and_show(fig, save_path, show_plot)


def plot_accuracy_comparison_only(
    comparison: Dict,
    save_path: Optional[str] = None,
    show_plot: bool = True
):
    """
    Plot only the accuracy comparison (single plot, matching paper's Figure 5).

    Parameters:
    -----------
    comparison : dict
        Comparison results
    save_path : str, optional
        Path to save the figure
    show_plot : bool, default=True
        If True, display the figure
    """

    if save_path is None:
        save_path = _get_default_save_path("accuracy_comparison.png")

    snr_values = comparison['snr']
    nn_acc = comparison['nn_accuracy']
    corr_acc = comparison['corr_accuracy']

    fig, ax = plt.subplots(figsize=config.FIGURE_SIZE_MEDIUM)

    ax.plot(snr_values, nn_acc, 'b-o', linewidth=2.5, markersize=10,
            label='Neural Network', markerfacecolor='white', markeredgewidth=2)
    ax.plot(snr_values, corr_acc, 'r--s', linewidth=2.5, markersize=10,
            label='Correlation Decoder', markerfacecolor='white', markeredgewidth=2)

    ax.axhline(y=99, color='green', linestyle=':', linewidth=2,
               label='3GPP Requirement (99%)', alpha=0.8)

    ax.set_xlabel('SNR (dB)', fontsize=config.FONT_SIZE_LARGE)
    ax.set_ylabel('Accuracy (%)', fontsize=config.FONT_SIZE_LARGE)
    ax.set_title('Model Accuracy', fontsize=config.FONT_SIZE_TITLE)
    ax.legend(fontsize=config.FONT_SIZE_SMALL, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(snr_values)

    min_acc = min(min(nn_acc), min(corr_acc))
    ax.set_ylim([max(0, min_acc - 5), 102])

    plt.tight_layout()

    _save_and_show(fig, save_path, show_plot)


# =============================================================================
# SECTION 7: PER-CLASS ACCURACY PLOT
# =============================================================================

def plot_per_class_accuracy(
    all_results: Dict[int, Dict],
    save_path: Optional[str] = None,
    show_plot: bool = True
):
    """
    Plot per-class accuracy across all SNR values.

    Parameters:
    -----------
    all_results : dict
        Results from evaluate_all_snr()
    save_path : str, optional
        Path to save the figure
    show_plot : bool, default=True
        If True, display the figure
    """

    if save_path is None:
        save_path = _get_default_save_path("per_class_accuracy.png")

    snr_values = sorted(all_results.keys())

    fig, ax = plt.subplots(figsize=config.FIGURE_SIZE_MEDIUM)

    # Colors and markers for each class
    colors = ['blue', 'red', 'green', 'orange']
    markers = ['o', 's', '^', 'D']

    for c in range(config.NUM_CLASSES):
        class_acc = [all_results[snr]['per_class_accuracy']
                     [c] * 100 for snr in snr_values]
        class_label = config.CLASS_LABELS.get(c, f"Class {c}")

        ax.plot(snr_values, class_acc,
                color=colors[c % len(colors)],
                marker=markers[c % len(markers)],
                linewidth=2,
                markersize=8,
                label=f'Class {c}: {class_label}',
                markerfacecolor='white',
                markeredgewidth=2)

    ax.set_xlabel('SNR (dB)', fontsize=config.FONT_SIZE_LARGE)
    ax.set_ylabel('Accuracy (%)', fontsize=config.FONT_SIZE_LARGE)
    ax.set_title('Per-Class Accuracy vs SNR', fontsize=config.FONT_SIZE_TITLE)
    ax.legend(fontsize=config.FONT_SIZE_SMALL, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(snr_values)

    min_acc = min([
        all_results[snr]['per_class_accuracy'][c] * 100
        for snr in snr_values
        for c in range(config.NUM_CLASSES)
    ])
    ax.set_ylim([max(0, min_acc - 5), 102])

    plt.tight_layout()

    _save_and_show(fig, save_path, show_plot)


# =============================================================================
# SECTION 8: GENERATE ALL PLOTS
# =============================================================================

def generate_all_plots(
    history: Dict,
    all_results: Dict[int, Dict],
    comparison: Optional[Dict] = None,
    show_plots: bool = True,
    verbose: bool = True
):
    """
    Generate all visualization plots.

    This function creates and saves all plots:
        1. Training history (accuracy + loss)
        2. Training accuracy only
        3. Training loss only
        4. Accuracy vs SNR
        5. Confusion matrices (all SNR)
        6. Confusion matrices (selected SNR)
        7. Individual confusion matrices (for key SNR values)
        8. Comparison plot (NN vs Correlation) - if comparison data available
        9. Accuracy comparison only - if comparison data available
        10. Per-class accuracy

    Parameters:
    -----------
    history : dict
        Training history
    all_results : dict
        Evaluation results for each SNR
    comparison : dict, optional
        Comparison results (NN vs Correlation)
    show_plots : bool, default=True
        If True, display all plots
    verbose : bool, default=True
        If True, print progress
    """

    if verbose:
        print("\n" + "=" * 70)
        print("GENERATING ALL PLOTS")
        print("=" * 70 + "\n")

    # 1. Training history
    if verbose:
        print("1. Training history (accuracy + loss)...")
    plot_training_history(
        history=history,
        save_path=_get_default_save_path("training_history.png"),
        show_plot=show_plots
    )

    # 2. Accuracy only
    if verbose:
        print("2. Training accuracy curve...")
    plot_accuracy_only(
        history=history,
        save_path=_get_default_save_path("accuracy_curve.png"),
        show_plot=show_plots
    )

    # 3. Loss only
    if verbose:
        print("3. Training loss curve...")
    plot_loss_only(
        history=history,
        save_path=_get_default_save_path("loss_curve.png"),
        show_plot=show_plots
    )

    # 4. Accuracy vs SNR
    if verbose:
        print("4. Accuracy vs SNR...")
    plot_accuracy_vs_snr(
        all_results=all_results,
        comparison=comparison,
        save_path=_get_default_save_path("accuracy_vs_snr.png"),
        show_plot=show_plots
    )

    # 5. All confusion matrices
    if verbose:
        print("5. Confusion matrices (all SNR)...")
    plot_confusion_matrices_all(
        all_results=all_results,
        save_path=_get_default_save_path("confusion_matrices_all.png"),
        show_plot=show_plots
    )

    # 6. Selected confusion matrices
    if verbose:
        print("6. Confusion matrices (selected SNR)...")
    plot_selected_confusion_matrices(
        all_results=all_results,
        save_path=_get_default_save_path("confusion_matrices_selected.png"),
        show_plot=show_plots
    )

    # 7. Individual confusion matrices for key SNR values
    snr_values = sorted(all_results.keys())
    key_snrs = [snr_values[0], snr_values[len(snr_values)//2], snr_values[-1]]

    for snr in key_snrs:
        if snr in all_results:
            if verbose:
                print(f"7. Individual confusion matrix (SNR = {snr} dB)...")
            plot_confusion_matrix_single(
                conf_matrix=all_results[snr]['confusion_matrix'],
                snr_db=snr,
                accuracy=all_results[snr]['accuracy'],
                save_path=_get_default_save_path(
                    f"confusion_matrix_{snr}dB.png"),
                show_plot=show_plots
            )

    # 8. Comparison plot
    if comparison is not None:
        if verbose:
            print("8. NN vs Correlation comparison...")
        plot_comparison(
            comparison=comparison,
            save_path=_get_default_save_path("nn_vs_correlation.png"),
            show_plot=show_plots
        )

        # 9. Accuracy comparison only
        if verbose:
            print("9. Accuracy comparison only...")
        plot_accuracy_comparison_only(
            comparison=comparison,
            save_path=_get_default_save_path("accuracy_comparison.png"),
            show_plot=show_plots
        )

    # 10. Per-class accuracy
    if verbose:
        print("10. Per-class accuracy...")
    plot_per_class_accuracy(
        all_results=all_results,
        save_path=_get_default_save_path("per_class_accuracy.png"),
        show_plot=show_plots
    )

    if verbose:
        print(f"\nAll plots saved to: {config.PLOTS_DIR}")
        print("=" * 70 + "\n")


# =============================================================================
# SECTION 9: SELF-TEST
# =============================================================================

if __name__ == "__main__":
    """
    Self-test for visualization module.
    """

    print("\n" + "=" * 70)
    print("VISUALIZATION MODULE - SELF TEST")
    print("=" * 70)

    # Create dummy data
    print("\n--- Creating dummy data for testing ---")

    np.random.seed(42)

    # Dummy training history
    num_epochs = 50
    dummy_history = {
        'accuracy': [0.25 + 0.7 * (1 - np.exp(-i/10)) + np.random.randn()*0.02 for i in range(num_epochs)],
        'val_accuracy': [0.25 + 0.68 * (1 - np.exp(-i/10)) + np.random.randn()*0.03 for i in range(num_epochs)],
        'loss': [1.4 * np.exp(-i/15) + 0.1 + np.random.randn()*0.02 for i in range(num_epochs)],
        'val_loss': [1.4 * np.exp(-i/15) + 0.15 + np.random.randn()*0.03 for i in range(num_epochs)]
    }

    # Clip accuracy to [0, 1]
    dummy_history['accuracy'] = [max(0, min(1, a))
                                 for a in dummy_history['accuracy']]
    dummy_history['val_accuracy'] = [
        max(0, min(1, a)) for a in dummy_history['val_accuracy']]
    dummy_history['loss'] = [max(0, l) for l in dummy_history['loss']]
    dummy_history['val_loss'] = [max(0, l) for l in dummy_history['val_loss']]

    # Dummy evaluation results
    snr_values = [0, 5, 10, 15, 20]
    dummy_all_results = {}

    for snr in snr_values:
        acc = 0.7 + 0.06 * snr + np.random.randn() * 0.01
        acc = max(0, min(1, acc))

        # Create confusion matrix
        n = 1000
        correct = int(n * acc)
        errors = n - correct

        cm = np.zeros((4, 4), dtype=int)
        for c in range(4):
            cm[c, c] = correct // 4

        # Distribute errors
        for _ in range(errors):
            r = np.random.randint(0, 4)
            c_wrong = np.random.randint(0, 3)
            if c_wrong >= r:
                c_wrong += 1
            cm[r, c_wrong] += 1

        per_class_acc = cm.diagonal() / cm.sum(axis=1)

        dummy_all_results[snr] = {
            'snr_db': snr,
            'accuracy': acc,
            'precision': acc - 0.01,
            'recall': acc - 0.005,
            'f1_score': acc - 0.008,
            'per_class_accuracy': per_class_acc,
            'confusion_matrix': cm,
            'y_true': np.random.randint(0, 4, n),
            'y_pred': np.random.randint(0, 4, n),
            'num_samples': n,
            'num_correct': correct,
            'num_errors': errors
        }

    # Dummy comparison
    dummy_comparison = {
        'snr': snr_values,
        'nn_accuracy': [r['accuracy'] * 100 for r in [dummy_all_results[s] for s in snr_values]],
        'corr_accuracy': [r['accuracy'] * 100 - 3 - np.random.rand()*2 for r in [dummy_all_results[s] for s in snr_values]]
    }

    print("Dummy data created successfully")

    # Create test directory
    test_dir = "./test_temp_plots/"
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    # Test 1: plot_training_history
    print("\n--- Test 1: plot_training_history ---")
    try:
        plot_training_history(
            history=dummy_history,
            save_path=os.path.join(test_dir, "test_history.png"),
            show_plot=False
        )
        print("Test 1: PASS")
    except Exception as e:
        print(f"Test 1: FAIL - {e}")

    # Test 2: plot_accuracy_only
    print("\n--- Test 2: plot_accuracy_only ---")
    try:
        plot_accuracy_only(
            history=dummy_history,
            save_path=os.path.join(test_dir, "test_acc.png"),
            show_plot=False
        )
        print("Test 2: PASS")
    except Exception as e:
        print(f"Test 2: FAIL - {e}")

    # Test 3: plot_loss_only
    print("\n--- Test 3: plot_loss_only ---")
    try:
        plot_loss_only(
            history=dummy_history,
            save_path=os.path.join(test_dir, "test_loss.png"),
            show_plot=False
        )
        print("Test 3: PASS")
    except Exception as e:
        print(f"Test 3: FAIL - {e}")

    # Test 4: plot_accuracy_vs_snr
    print("\n--- Test 4: plot_accuracy_vs_snr ---")
    try:
        plot_accuracy_vs_snr(
            all_results=dummy_all_results,
            comparison=dummy_comparison,
            save_path=os.path.join(test_dir, "test_acc_snr.png"),
            show_plot=False
        )
        print("Test 4: PASS")
    except Exception as e:
        print(f"Test 4: FAIL - {e}")

    # Test 5: plot_confusion_matrices_all
    print("\n--- Test 5: plot_confusion_matrices_all ---")
    try:
        plot_confusion_matrices_all(
            all_results=dummy_all_results,
            save_path=os.path.join(test_dir, "test_cm_all.png"),
            show_plot=False
        )
        print("Test 5: PASS")
    except Exception as e:
        print(f"Test 5: FAIL - {e}")

    # Test 6: plot_confusion_matrix_single
    print("\n--- Test 6: plot_confusion_matrix_single ---")
    try:
        plot_confusion_matrix_single(
            conf_matrix=dummy_all_results[10]['confusion_matrix'],
            snr_db=10,
            accuracy=dummy_all_results[10]['accuracy'],
            save_path=os.path.join(test_dir, "test_cm_single.png"),
            show_plot=False
        )
        print("Test 6: PASS")
    except Exception as e:
        print(f"Test 6: FAIL - {e}")

    # Test 7: plot_selected_confusion_matrices
    print("\n--- Test 7: plot_selected_confusion_matrices ---")
    try:
        plot_selected_confusion_matrices(
            all_results=dummy_all_results,
            selected_snrs=[0, 10, 20],
            save_path=os.path.join(test_dir, "test_cm_sel.png"),
            show_plot=False
        )
        print("Test 7: PASS")
    except Exception as e:
        print(f"Test 7: FAIL - {e}")

    # Test 8: plot_comparison
    print("\n--- Test 8: plot_comparison ---")
    try:
        plot_comparison(
            comparison=dummy_comparison,
            save_path=os.path.join(test_dir, "test_comp.png"),
            show_plot=False
        )
        print("Test 8: PASS")
    except Exception as e:
        print(f"Test 8: FAIL - {e}")

    # Test 9: plot_accuracy_comparison_only
    print("\n--- Test 9: plot_accuracy_comparison_only ---")
    try:
        plot_accuracy_comparison_only(
            comparison=dummy_comparison,
            save_path=os.path.join(test_dir, "test_acc_comp.png"),
            show_plot=False
        )
        print("Test 9: PASS")
    except Exception as e:
        print(f"Test 9: FAIL - {e}")

    # Test 10: plot_per_class_accuracy
    print("\n--- Test 10: plot_per_class_accuracy ---")
    try:
        plot_per_class_accuracy(
            all_results=dummy_all_results,
            save_path=os.path.join(test_dir, "test_per_class.png"),
            show_plot=False
        )
        print("Test 10: PASS")
    except Exception as e:
        print(f"Test 10: FAIL - {e}")

    # Test 11: generate_all_plots
    print("\n--- Test 11: generate_all_plots ---")
    try:
        # Temporarily change plots dir
        original_plots_dir = config.PLOTS_DIR
        config.PLOTS_DIR = test_dir

        generate_all_plots(
            history=dummy_history,
            all_results=dummy_all_results,
            comparison=dummy_comparison,
            show_plots=False,
            verbose=True
        )

        # Restore
        config.PLOTS_DIR = original_plots_dir
        print("Test 11: PASS")
    except Exception as e:
        config.PLOTS_DIR = original_plots_dir
        print(f"Test 11: FAIL - {e}")

    # Cleanup test files
    print("\n--- Cleaning up test files ---")
    try:
        import glob
        test_files = glob.glob(os.path.join(test_dir, "*.png"))
        for f in test_files:
            os.remove(f)
        os.rmdir(test_dir)
        print(f"Cleaned up {len(test_files)} test files")
    except Exception as e:
        print(f"Cleanup warning: {e}")

    print("\n" + "=" * 70)
    print("SELF TEST COMPLETE")
    print("=" * 70)
