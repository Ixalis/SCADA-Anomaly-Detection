# SCADA/ICS Anomaly Detection System

A machine learning-based anomaly detection system for industrial SCADA (Supervisory Control and Data Acquisition) systems using the SWaT (Secure Water Treatment) dataset.

**Project Title:** Xay dung he thong tu dong kiem tra tin hieu SCADA/ICS (Building an Automatic SCADA/ICS Signal Inspection System)

---

## Table of Contents

1. [Overview](#overview)
2. [Repository Structure](#repository-structure)
3. [Dataset Setup](#dataset-setup)
4. [How to Use](#how-to-use)
5. [Notebooks Description](#notebooks-description)
6. [Results Summary](#results-summary)
7. [References](#references)

---

## Overview

This project implements multiple machine learning approaches for detecting anomalies and cyber attacks in industrial control systems. The system compares supervised learning (Random Forest), unsupervised learning (Isolation Forest), and deep learning (LSTM-Autoencoder) methods.

### Key Features

- Multi-model comparison framework (5 algorithms)
- Multi-type anomaly detection (7 anomaly categories)
- Hybrid cascade detection pipeline
- LSTM-Autoencoder for temporal pattern recognition

---

## Repository Structure

```
SCADA-Anomaly-Detection/
│
├── notebooks/
│   ├── 00_Convert_Excel_to_CSV.ipynb
│   ├── 01_Train_IsolationForest_SWaT.ipynb
│   ├── 02_Inference_IsolationForest_SWaT.ipynb
│   ├── Multi_Model_Comparison_SCADA.ipynb
│   ├── Multi_Type_Anomaly_Detection_SCADA.ipynb
│   ├── LSTM_Autoencoder_SCADA.ipynb
│   └── Hybrid_Anomaly_Detection_SCADA.ipynb
│
├── data/                        # Dataset folder (NOT included)
│   └── .gitkeep
│
├── README.md
└── requirements.txt
```

---

## Dataset Setup

### IMPORTANT: You Must Download the Dataset Yourself

The SWaT (Secure Water Treatment) dataset is proprietary and maintained by iTrust, Centre for Research in Cyber Security, Singapore University of Technology and Design.

**The dataset is NOT included in this repository.**

### How to Obtain the Dataset

1. Visit the iTrust dataset request page:
   ```
   https://itrust.sutd.edu.sg/itrust-labs_datasets/
   ```

2. Submit a request form with your institutional affiliation and intended use

3. Upon approval, download:
   - `SWaT_Dataset_Normal_v1.xlsx` (Normal operation data)
   - `SWaT_Dataset_Attack_v0.xlsx` (Attack data for testing)

4. Place the files in your Google Drive

5. Run `00_Convert_Excel_to_CSV.ipynb` to convert to CSV format

---

## How to Use

### Prerequisites

- Google account (for Google Colab)
- SWaT dataset (see Dataset Setup above)

### Step-by-Step Instructions

**Step 1: Upload notebooks to Google Colab**

Download the notebooks from this repository and upload them to Google Colab, or open them directly from GitHub.

**Step 2: Mount Google Drive**

Each notebook starts with:
```python
from google.colab import drive
drive.mount('/content/drive')
```

**Step 3: Update data paths**

In each notebook, update the paths to match your Google Drive folder:
```python
NORMAL_DATA_PATH = '/content/drive/MyDrive/your_folder/SWaT_Normal.csv'
ATTACK_DATA_PATH = '/content/drive/MyDrive/your_folder/SWaT_Attack.csv'
MODEL_SAVE_PATH = '/content/drive/MyDrive/your_folder/models/'
```

**Step 4: Run notebooks in order**

| Order | Notebook | Purpose |
|-------|----------|---------|
| 1 | `00_Convert_Excel_to_CSV.ipynb` | Convert Excel to CSV (run once) |
| 2 | `01_Train_IsolationForest_SWaT.ipynb` | Train Isolation Forest model |
| 3 | `02_Inference_IsolationForest_SWaT.ipynb` | Test predictions |
| 4 | `Multi_Model_Comparison_SCADA.ipynb` | Compare all 5 models |
| 5 | `Multi_Type_Anomaly_Detection_SCADA.ipynb` | Detect 7 anomaly types |
| 6 | `LSTM_Autoencoder_SCADA.ipynb` | Train deep learning model |
| 7 | `Hybrid_Anomaly_Detection_SCADA.ipynb` | Run hybrid pipeline |

**Step 5: For LSTM notebook, enable GPU**

Go to Runtime -> Change runtime type -> Hardware accelerator -> GPU

---

## Notebooks Description

### Data Preprocessing

**00_Convert_Excel_to_CSV.ipynb**
- Converts Excel files from iTrust to CSV format
- Run this first before any other notebook

### Model Training and Inference

**01_Train_IsolationForest_SWaT.ipynb**
- Trains Isolation Forest on normal data only
- Saves model and scaler files

**02_Inference_IsolationForest_SWaT.ipynb**
- Loads trained model
- Runs predictions on attack data
- Outputs performance metrics

### Analysis Notebooks

**Multi_Model_Comparison_SCADA.ipynb**

Compares 5 algorithms side-by-side:
- Random Forest (supervised)
- Isolation Forest (unsupervised)
- One-Class SVM (unsupervised)
- Local Outlier Factor (unsupervised)
- Autoencoder (deep learning)

**Multi_Type_Anomaly_Detection_SCADA.ipynb**

Detects 7 types of anomalies:
- Stuck Sensor
- Sensor Drift
- Noise Burst
- Out-of-Range
- Spike
- Data Drop
- Correlation Break

**LSTM_Autoencoder_SCADA.ipynb**
- Builds LSTM-Autoencoder for sequence-based detection
- Uses sliding window (30 timesteps)
- Requires GPU for faster training

**Hybrid_Anomaly_Detection_SCADA.ipynb**
- Implements three-layer cascade: Rule-based -> Isolation Forest -> Random Forest
- Demonstrates the complete detection pipeline

---

## Results Summary

### Model Performance Comparison (SWaT Dataset)

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Random Forest | 99.85% | 99.77% | 99.99% | 0.9988 |
| Isolation Forest | 54.90% | 47.30% | 99.80% | 0.5490 |
| One-Class SVM | 63.21% | 52.10% | 88.50% | 0.6321 |
| LOF | 66.82% | 54.80% | 91.20% | 0.6682 |
| LSTM-Autoencoder | 33.97% | 15.10% | 93.93% | 0.2601 |

### Key Findings

- **Random Forest** achieves the best performance when labeled data is available
- **Isolation Forest** is ideal for deployments without labeled attack data
- **LSTM-Autoencoder** achieves high recall (93.9%), suitable when missing attacks is costly

---

## References

1. Goh, J., et al. (2016). A Dataset to Support Research in the Design of Secure Water Treatment Systems. CRITIS 2016.

2. iTrust, Centre for Research in Cyber Security, SUTD. SWaT Dataset. https://itrust.sutd.edu.sg/

3. Liu, F. T., Ting, K. M., and Zhou, Z. H. (2008). Isolation Forest. ICDM 2008.

---

## License

This project is for educational and research purposes.

---

## Acknowledgments

- iTrust, Singapore University of Technology and Design for the SWaT dataset
- Ho Chi Minh City University of Technology (HCMUT-VNUHCM)
