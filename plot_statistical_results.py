"""
Plot Statistical Results with Error Bars
For research publication
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load statistical summary
df = pd.read_csv("./results_multi_ue/statistical/statistical_summary.csv")

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: NN Accuracy with 95% CI
snr = df['SNR_dB'].values
nn_mean = df['NN_Accuracy_Mean'].values
nn_std = df['NN_Accuracy_Std'].values
nn_ci_lower = df['NN_Accuracy_95CI_Lower'].values
nn_ci_upper = df['NN_Accuracy_95CI_Upper'].values

corr_mean = df['Corr_Accuracy_Mean'].values
corr_std = df['Corr_Accuracy_Std'].values

# Plot NN with error bars
axes[0].errorbar(snr, nn_mean, yerr=[nn_mean - nn_ci_lower, nn_ci_upper - nn_mean],
                 fmt='bo-', linewidth=2.5, markersize=10, capsize=5,
                 label='Neural Network (95% CI)', markerfacecolor='white', markeredgewidth=2)

# Plot Correlation (no error bars for baseline)
axes[0].plot(snr, corr_mean, 'rs--', linewidth=2.5, markersize=10,
             label='Correlation Decoder', markerfacecolor='white', markeredgewidth=2)

axes[0].axhline(y=99, color='gray', linestyle=':', linewidth=1.5, alpha=0.7,
                label='3GPP Requirement (99%)')
axes[0].set_xlabel('SNR (dB)', fontsize=14)
axes[0].set_ylabel('Accuracy (%)', fontsize=14)
axes[0].set_title(
    'Neural Network vs Correlation Decoder\n(with 95% Confidence Intervals)', fontsize=16)
axes[0].legend(fontsize=11, loc='lower right')
axes[0].grid(True, alpha=0.3)
axes[0].set_xticks(snr)
axes[0].set_ylim([0, 102])

# Right: Accuracy Gain
gain_mean = df['Gain_Mean'].values
gain_std = df['Gain_Std'].values

colors = ['green' if g > 40 else 'orange' for g in gain_mean]
bars = axes[1].bar(snr, gain_mean, yerr=gain_std, capsize=5,
                   color=colors, edgecolor='black', linewidth=1.5)

# Add value labels on bars
for bar, gain in zip(bars, gain_mean):
    height = bar.get_height()
    axes[1].annotate(f'+{gain:.1f}%',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=11, fontweight='bold')

axes[1].set_xlabel('SNR (dB)', fontsize=14)
axes[1].set_ylabel('Accuracy Gain (%)', fontsize=14)
axes[1].set_title('Neural Network Gain Over Correlation Decoder', fontsize=16)
axes[1].grid(True, alpha=0.3, axis='y')
axes[1].set_xticks(snr)
axes[1].set_ylim([0, 55])

plt.tight_layout()

# Save
save_path = "./results_multi_ue/statistical/final_results_with_ci.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to: {save_path}")

plt.show()
