# SCADA/ICS Anomaly Detection System

A machine learning-based anomaly detection system for industrial SCADA (Supervisory Control and Data Acquisition) systems using the SWaT (Secure Water Treatment) dataset.

**Project Title:** Xay dung he thong tu dong kiem tra tin hieu SCADA/ICS (Building an Automatic SCADA/ICS Signal Inspection System)

---

## Table of Contents

1. [Overview](#overview)
2. [Repository Structure](#repository-structure)
3. [Prerequisites](#prerequisites)
4. [Dataset Setup](#dataset-setup)
5. [Pipeline Flow](#pipeline-flow)
6. [Notebooks Description](#notebooks-description)
7. [Models Implemented](#models-implemented)
8. [Usage Instructions](#usage-instructions)
9. [Results Summary](#results-summary)
10. [References](#references)

---

## Overview

This project implements multiple machine learning approaches for detecting anomalies and cyber attacks in industrial control systems. The system uses a hybrid cascade architecture combining rule-based detection, unsupervised learning (Isolation Forest), supervised learning (Random Forest), and deep learning (LSTM-Autoencoder).

### Key Features

- Multi-model comparison framework (5 algorithms)
- Multi-type anomaly detection (7 anomaly categories)
- Hybrid cascade detection pipeline
- LSTM-Autoencoder for temporal pattern recognition
- Real-time detection capability

---

## Repository Structure

```
SCADA-Anomaly-Detection/
│
├── notebooks/
│   ├── 00_Convert_Excel_to_CSV.ipynb      # Data preprocessing
│   ├── 01_Train_IsolationForest_SWaT.ipynb
│   ├── 02_Inference_IsolationForest_SWaT.ipynb
│   ├── Multi_Model_Comparison_SCADA.ipynb
│   ├── Multi_Type_Anomaly_Detection_SCADA.ipynb
│   ├── LSTM_Autoencoder_SCADA.ipynb
│   └── Hybrid_Anomaly_Detection_SCADA.ipynb
│
├── SCADA_SIM_Enhanced/                     # Simulator (optional)
│   ├── config/
│   │   └── config.yaml
│   ├── src/
│   │   ├── data_simulator.py
│   │   ├── lstm_autoencoder_detector.py
│   │   └── utils.py
│   ├── models/                             # Trained models stored here
│   ├── main.py
│   └── requirements.txt
│
├── models/                                 # Saved model files
│   ├── isolation_forest_model.pkl
│   ├── random_forest_model.pkl
│   ├── lstm_autoencoder.keras
│   ├── lstm_ae_scaler.pkl
│   ├── lstm_ae_config.pkl
│   └── scaler.pkl
│
├── data/                                   # Dataset folder (NOT included)
│   └── .gitkeep
│
├── results/                                # Output results
│   ├── model_comparison_results.csv
│   └── anomaly_type_summary.csv
│
├── README.md
└── requirements.txt
```

---

## Prerequisites

### Software Requirements

- Python 3.8 or higher
- Google Colab (recommended) or Jupyter Notebook
- 12GB+ RAM for full dataset processing

### Python Dependencies

```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
tensorflow>=2.10.0
matplotlib>=3.4.0
seaborn>=0.11.0
joblib>=1.1.0
pyyaml>=6.0
openpyxl>=3.0.0
```

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Dataset Setup

### IMPORTANT: You Must Download the Dataset Yourself

The SWaT (Secure Water Treatment) dataset is proprietary and maintained by iTrust, Centre for Research in Cyber Security, Singapore University of Technology and Design.

**The dataset is NOT included in this repository. You must obtain it directly from iTrust.**

### How to Obtain the Dataset

1. Visit the iTrust dataset request page:
   ```
   https://itrust.sutd.edu.sg/itrust-labs_datasets/
   ```

2. Submit a request form with your institutional affiliation and intended use

3. Upon approval, download the following files:
   - `SWaT_Dataset_Normal_v1.xlsx` (Normal operation data)
   - `SWaT_Dataset_Attack_v0.xlsx` (Attack data for testing)

4. Place the files in the `data/` directory

5. Run the conversion notebook to generate CSV files:
   ```
   notebooks/00_Convert_Excel_to_CSV.ipynb
   ```

### Expected Data Format

After conversion, your `data/` folder should contain:

```
data/
├── SWaT_Normal.csv      # ~495,000 samples of normal operation
└── SWaT_Attack.csv      # ~449,000 samples including attack periods
```

### Dataset Structure

| Column Type | Examples | Count |
|-------------|----------|-------|
| Timestamp | Timestamp | 1 |
| Sensors (Analog) | LIT101, FIT101, AIT201, etc. | 25 |
| Actuators (Digital) | P101, MV101, P201, etc. | 26 |
| Label | Normal/Attack | 1 |

---

## Pipeline Flow

The detection system follows a cascaded architecture:

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA INGESTION                               │
│         Load SWaT CSV → Preprocess → Scale Features             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              LAYER 0: RULE-BASED DETECTION                      │
│                                                                 │
│    Check physical limits (e.g., LIT101 must be 0-1000mm)        │
│    Latency: <1ms                                                │
│                                                                 │
│    [PASS] ──────────────────────────────────────────────────────┼──► NORMAL
│    [VIOLATION] ─────────────────────────────────────────────────┼──► ALERT
│    [UNCERTAIN]                                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│           LAYER 1: ISOLATION FOREST (Unsupervised)              │
│                                                                 │
│    Trained on normal data only                                  │
│    Detects statistical anomalies without labels                 │
│    Latency: ~10ms                                               │
│                                                                 │
│    [NORMAL] ────────────────────────────────────────────────────┼──► NORMAL
│    [ANOMALY DETECTED]                                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│           LAYER 2: RANDOM FOREST (Supervised)                   │
│                                                                 │
│    Only runs when Layer 1 flags anomaly                         │
│    Classifies attack type (if labels available)                 │
│    Latency: ~5ms                                                │
│                                                                 │
│    Output: Attack classification + confidence score             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      ALERT OUTPUT                               │
│         Detection result + Attack type + Confidence             │
└─────────────────────────────────────────────────────────────────┘
```

### Why This Hybrid Approach?

| Layer | Strength | Purpose |
|-------|----------|---------|
| Rule-based | Zero false negatives for limit violations | Catch obvious attacks instantly |
| Isolation Forest | No labels required, detects novel attacks | First-line anomaly detection |
| Random Forest | High accuracy when labels available | Attack classification |

---

## Notebooks Description

### 1. Data Preprocessing

**00_Convert_Excel_to_CSV.ipynb**
- Converts Excel files from iTrust to CSV format
- Handles encoding issues and column name cleaning
- Run this first before any other notebook

### 2. Model Training

**01_Train_IsolationForest_SWaT.ipynb**
- Trains Isolation Forest on normal operation data only
- Saves model, scaler, and configuration files
- Outputs: `isolation_forest_model.pkl`, `scaler.pkl`

**LSTM_Autoencoder_SCADA.ipynb**
- Builds and trains LSTM-Autoencoder for sequence-based detection
- Creates sliding window sequences (30 timesteps)
- Outputs: `lstm_autoencoder.keras`, `lstm_ae_scaler.pkl`, `lstm_ae_config.pkl`

### 3. Model Comparison

**Multi_Model_Comparison_SCADA.ipynb**
- Compares 5 algorithms side-by-side:
  - Random Forest (supervised)
  - Isolation Forest (unsupervised)
  - One-Class SVM (unsupervised)
  - Local Outlier Factor (unsupervised)
  - Autoencoder (deep learning)
- Generates performance metrics, ROC curves, and confusion matrices
- Outputs: `model_comparison_results.csv`

### 4. Anomaly Analysis

**Multi_Type_Anomaly_Detection_SCADA.ipynb**
- Detects 7 types of anomalies beyond simple attack classification:

| Anomaly Type | Description | Potential Cause |
|--------------|-------------|-----------------|
| Stuck Sensor | Constant values over time | Sensor failure, replay attack |
| Sensor Drift | Gradual deviation from baseline | Calibration issues, stealthy attack |
| Noise Burst | Sudden increase in variability | EMI, electrical interference |
| Out-of-Range | Values exceed physical limits | Sensor malfunction, attack |
| Spike | Sudden sharp changes | Process disturbance, attack |
| Data Drop | Missing or null values | Network issues, sensor failure |
| Correlation Break | Related sensors stop correlating | Targeted attack |

### 5. Hybrid Detection

**Hybrid_Anomaly_Detection_SCADA.ipynb**
- Implements the three-layer cascade architecture
- Combines rule-based, Isolation Forest, and Random Forest
- Demonstrates real-time detection flow

### 6. Inference

**02_Inference_IsolationForest_SWaT.ipynb**
- Loads pre-trained models
- Runs predictions on test data
- Evaluates detection performance

---

## Models Implemented

### Supervised Methods

| Model | Training Data | Strengths |
|-------|---------------|-----------|
| Random Forest | Labeled (normal + attack) | Highest accuracy (F1: 0.9988), interpretable |

### Unsupervised Methods

| Model | Training Data | Strengths |
|-------|---------------|-----------|
| Isolation Forest | Normal only | Fast, no labels needed, detects novel attacks |
| One-Class SVM | Normal only | Good for well-defined normal boundary |
| Local Outlier Factor | Normal only | Density-based, good for local anomalies |

### Deep Learning Methods

| Model | Training Data | Strengths |
|-------|---------------|-----------|
| LSTM-Autoencoder | Normal sequences | Captures temporal patterns, high recall (93.9%) |
| Standard Autoencoder | Normal only | Simpler, faster training |

---

## Usage Instructions

### Running on Google Colab (Recommended)

1. Upload the notebooks to Google Colab

2. Mount your Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

3. Update the data paths in each notebook:
   ```python
   NORMAL_DATA_PATH = '/content/drive/MyDrive/your_folder/SWaT_Normal.csv'
   ATTACK_DATA_PATH = '/content/drive/MyDrive/your_folder/SWaT_Attack.csv'
   MODEL_SAVE_PATH = '/content/drive/MyDrive/your_folder/models/'
   ```

4. Run notebooks in order:
   ```
   00_Convert_Excel_to_CSV.ipynb  →  First (if you have Excel files)
   01_Train_IsolationForest_SWaT.ipynb  →  Training
   Multi_Model_Comparison_SCADA.ipynb  →  Compare all models
   02_Inference_IsolationForest_SWaT.ipynb  →  Test predictions
   ```

### Running Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/SCADA-Anomaly-Detection.git
   cd SCADA-Anomaly-Detection
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Place SWaT dataset in `data/` folder

4. Run Jupyter:
   ```bash
   jupyter notebook
   ```

Note: Local execution requires 12GB+ RAM for full dataset. Use the memory-optimized settings in notebooks if running on limited hardware.

---

## Results Summary

### Model Performance Comparison (SWaT Dataset)

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| Random Forest | 99.85% | 99.77% | 99.99% | 0.9988 | 0.9999 |
| Isolation Forest | 54.90% | 47.30% | 99.80% | 0.5490 | 0.7842 |
| One-Class SVM | 63.21% | 52.10% | 88.50% | 0.6321 | 0.7156 |
| LOF | 66.82% | 54.80% | 91.20% | 0.6682 | 0.7523 |
| LSTM-Autoencoder | 33.97% | 15.10% | 93.93% | 0.2601 | 0.7228 |

### Key Findings

1. **Random Forest** achieves the best overall performance when labeled data is available

2. **Isolation Forest** provides the best balance between detection rate and speed for unsupervised scenarios, making it ideal for deployments without labeled attack data

3. **LSTM-Autoencoder** achieves high recall (93.9%), catching most attacks at the cost of more false positives, suitable when missing attacks is more costly than false alarms

4. The **hybrid cascade approach** (Rule-based, Isolation Forest, Random Forest) provides the best practical deployment strategy

---

## References

1. Goh, J., et al. (2016). A Dataset to Support Research in the Design of Secure Water Treatment Systems. CRITIS 2016.

2. iTrust, Centre for Research in Cyber Security, SUTD. SWaT Dataset. https://itrust.sutd.edu.sg/

3. Liu, F. T., Ting, K. M., and Zhou, Z. H. (2008). Isolation Forest. ICDM 2008.

---

## License

This project is for educational and research purposes. The SWaT dataset has its own usage terms defined by iTrust.

---

## Acknowledgments

- iTrust, Singapore University of Technology and Design for the SWaT dataset
- Ho Chi Minh City University of Technology (HCMUT-VNUHCM)
