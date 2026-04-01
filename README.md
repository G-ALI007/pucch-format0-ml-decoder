# 🎯 PUCCH Format 0 ML Decoder

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.21+](https://img.shields.io/badge/tensorflow-2.21+-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper Status](https://img.shields.io/badge/paper-in__preparation-red.svg)]()
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXX.svg)]()

> **Research-Grade Implementation for 5G NR PUCCH Format 0 Decoding using Machine Learning**

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Contributions](#key-contributions)
- [Performance Results](#performance-results)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

---

## 📖 Overview

This repository implements a **Machine Learning-based decoder** for **5G New Radio (NR) PUCCH Format 0** that demonstrates **robust performance under multi-user interference**.

### Problem Statement

In dense 5G networks, multiple User Equipments (UEs) may transmit PUCCH Format 0 simultaneously on the same time-frequency resources using different cyclic shifts. This creates **inter-user interference** that degrades traditional correlation-based decoders.

### Proposed Solution

We train a **Fully Connected Neural Network** to decode the target user's UCI (ACK/NACK + SR) in the presence of interference from 1-2 other users. The model is trained on a single SNR (10 dB) and generalizes across a wide SNR range (0-20 dB).

---

## 🏆 Key Contributions

| # | Contribution | Impact |
|---|--------------|--------|
| 1 | **Multi-User Interference Robustness** | Decoder works with 2-3 simultaneous users |
| 2 | **45% Performance Gain** | Over traditional correlation-based decoder |
| 3 | **Statistical Validation** | 5 independent runs with 95% confidence intervals |
| 4 | **DTX Detection** | Extended 5-class model with Discontinuous Transmission detection |
| 5 | **Complete Reproducibility** | MATLAB data generation + Python ML pipeline |
| 6 | **Publication-Ready** | All results, plots, and analysis included |

---

## 📊 Performance Results

### Accuracy vs SNR (Multi-User Scenario)

| SNR (dB) | Neural Network | Correlation Decoder | **Gain** |
|----------|---------------|---------------------|----------|
| 0 | 64.7% ± 1.2% | 41.5% | **+23.2%** |
| 5 | 88.0% ± 0.9% | 48.0% | **+40.0%** |
| 10 | 97.4% ± 0.3% | 51.2% | **+46.2%** ⭐ |
| 15 | 98.1% ± 0.2% | 52.6% | **+45.5%** ⭐ |
| 20 | 98.2% ± 0.2% | 53.1% | **+45.1%** ⭐ |

### Key Metrics

- **Maximum Accuracy:** 98.2% (SNR ≥ 15 dB)
- **Average Gain:** +40.1% over correlation decoder
- **3GPP Gap:** < 1% from 99% requirement (at SNR ≥ 15 dB)
- **Statistical Confidence:** 95% CI width < 2%

### Training Performance
Best Validation Accuracy: 97.36% (Epoch 300)
Training Time: ~15 minutes (CPU)
Model Size: 134 KB
Inference Time: 0.15 ms/sample

---

## 🚀 Installation

### Prerequisites

- Python 3.12+
- MATLAB R2023a+ (for data generation)
- 10 GB free disk space

### Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/pucch-format0-ml-decoder.git
cd pucch-format0-ml-decoder
Create Virtual Environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
Install Dependencies
pip install -r requirements.txt
Requirements
tensorflow>=2.21.0
tf_keras>=2.15.0
numpy>=1.26.0
pandas>=2.2.0
scikit-learn>=1.4.0
matplotlib>=3.8.0
seaborn>=0.13.0
⚡ Quick Start
1. Generate Data (MATLAB)
% Single-User
run('matlab/generate_pucch_f0_data.m')

% Multi-User (2 & 3 users)
run('matlab/generate_pucch_f0_multiuser.m')

% DTX Detection
run('matlab/generate_pucch_f0_dtx.m')
2. Run Single-User Pipeline
python main.py
3. Run Multi-User Pipeline
python main_multi_ue.py
4. Run Statistical Validation (5 runs)
python run_multi_experiments_multi_ue.py --runs 5
5. Run DTX Detection Pipeline
python main_dtx.py
📁 Project Structure
pucch-format0-ml-decoder/
│
├── 📄 README.md                    # This file
├── 📄 LICENSE                      # MIT License
├── 📄 requirements.txt             # Python dependencies
├               
│
├── 📁 matlab/                      # MATLAB data generation
│   ├── generate_pucch_f0_data.m           # Single-user
│   ├── generate_pucch_f0_multiuser.m      # Multi-user (2 & 3 users)
│   └── generate_pucch_f0_dtx.m            # DTX detection
│
├── 📁 python/                      # Python ML pipeline
│   ├── config.py                   # Base configuration
│   ├── config_multi_ue.py          # Multi-user configuration
│   ├── config_dtx.py               # DTX configuration
│   ├── data_loader.py              # Data loading
│   ├── data_loader_multi_ue.py     # Multi-user data loading
│   ├── data_preprocessing.py       # Preprocessing
│   ├── data_preprocessing_multi_ue.py
│   ├── model.py                    # Neural network model
│   ├── evaluation.py               # Evaluation metrics
│   ├── evaluation_multi_ue.py      # Multi-user evaluation
│   ├── visualization.py            # Plotting
│   ├── main.py                     # Single-user pipeline
│   ├── main_multi_ue.py            # Multi-user pipeline
│   ├── main_dtx.py                 # DTX pipeline
│   ├── main_twostage.py            # Two-stage DTX
│   ├── main_architectures.py       # Architecture comparison
│   └── run_multi_experiments_multi_ue.py  # Statistical validation
│
├── 📁 results/                     # Single-user results
├── 📁 results_multi_ue/            # Multi-user results
├── 📁 results_dtx/                 # DTX results
├── 📁 models/                      # Trained models
├── 📁 plots/                       # Generated plots
└── 📁 logs/                        # Training logs
📖 Usage
Single-User Mode
# Full pipeline
python main.py

# Evaluation only (requires trained model)
python main.py --eval-only

# Without plots (batch mode)
python main.py --no-plots
Multi-User Mode
# Full pipeline (2 users)
python main_multi_ue.py

# 3 users (edit config_multi_ue.py first)
# CURRENT_SCENARIO_KEY = "3users"
python main_multi_ue.py

# Statistical validation (5 runs)
python run_multi_experiments_multi_ue.py --runs 5
DTX Detection Mode
# Full DTX pipeline
python main_dtx.py

# Two-stage approach
python main_twostage.py
Architecture Comparison
# Compare different neural network architectures
python main_architectures.py
📊 Results
All results are saved in:
results_multi_ue/statistical/statistical_summary.csv - Mean ± Std Dev
results_multi_ue/statistical/results_table.tex - LaTeX table for paper
results_multi_ue/plots/ - Publication-ready figures
Example Output
--- Multi-User Results Summary ---
SNR (dB)    NN Acc (%)  Corr Acc (%)  Gain (%)
--------------------------------------------------
0           64.70       41.53         +23.17
5           88.00       47.58         +40.42
10          97.40       51.17         +46.23
15          98.10       52.55         +45.55
20          98.20       53.08         +45.12
--------------------------------------------------
Average Gain: +40.10%
📚 Citation
If you use this code in your research, please cite:
@misc{pucch-ml-decoder-2024,
  author = {Ghader Ali},
  title = {PUCCH Format 0 ML Decoder: Multi-User Interference Robust Decoding for 5G NR},
  year = {2024},
  howpublished = {\url{https://github.com/G-ALI007/pucch-format0-ml-decoder}},
  doi = {10.5281/zenodo.XXXXXX}
}
🤝 Contributing
Contributions are welcome! Please follow these steps:
Fork the repository
Create a feature branch (git checkout -b feature/amazing-feature)
Commit your changes (git commit -m 'Add amazing feature')
Push to the branch (git push origin feature/amazing-feature)
Open a Pull Request
Code Style
Follow PEP 8 for Python code
Use type hints for all functions
Include docstrings for all public functions
Add tests for new features
📧 Contact
Author: ghader ali
Email: aghader563@gmail.com
🙏 Acknowledgments
3GPP for PUCCH Format 0 specifications
MATLAB Communications Toolbox for data generation
TensorFlow/Keras for neural network implementation
[Paper Base Reference] for baseline architecture
📌 Disclaimer
This code is for research and educational purposes. It implements the methodology described in our paper but may require modifications for production deployment.
<div align="center">

Made with ❤️ for 5G Research Community
⬆ Back to Top
</div>
```
