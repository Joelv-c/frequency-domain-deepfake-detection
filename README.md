# Frequency-Domain Deepfake Video Detection

## Overview
This repository contains the code and research for a deepfake video detection pipeline explicitly optimized for low-latency forensic triage. 

While most conventional detectors operate in the spatial domain and rely on pixel-level artifacts, this project explores a frequency-domain approach. By leveraging log-magnitude Fast Fourier Transform (FFT) spectra, the model achieves high accuracy and near-real-time inference speeds, making it highly viable for rapid triage of suspicious content.

You can read the full methodology, detailed mathematical formulations, and view the training graphs in the included `Capstone Project Report.pdf`.

## Architecture & Pipeline
The project evaluates a three-stage progression of models:

1. **Baseline 3D CNN:** Trained on short RGB video clips to capture spatial-temporal features.
2. **Intermediate ResNet50:** Leverages transfer learning applied to log-magnitude FFT spectra.
3. **Fine-Tuned ResNet50:** The final model, with the top convolutional block fine-tuned specifically for frequency-domain features.

### Tech Stack
* **Language:** Python
* **Framework:** PyTorch / Torchvision
* **Data Processing:** OpenCV (cv2), NumPy
* **Compute:** NVIDIA Tesla T4 GPU (for latency benchmarking)

## Performance & Results
The shift from the spatial domain to the frequency domain yielded significant performance improvements across six independent runs. *(Note: Full ROC/AUC curves and latency distribution graphs are available in the attached Capstone Project Report PDF).*

| Model Architecture | Accuracy | AUC Score | Median Latency (per-frame) |
| :--- | :--- | :--- | :--- |
| Baseline 3D CNN | 57% - 66% | 0.55 - 0.74 | N/A |
| FFT-based ResNet50 | 72% - 76% | 0.82 - 0.85 | N/A |
| **Fine-Tuned FFT ResNet50** | **90% - 92%** | **> 0.95** | **~2.7 ms** |

## Dataset & Acknowledgements
The video data used to train and evaluate these models is sourced from the **FaceForensics++** dataset. 
* Original Dataset Repository: [ondyari/FaceForensics](https://github.com/ondyari/FaceForensics)

## Repository Structure
```text
├── .gitignore
├── README.md
├── Capstone Project Report.pdf       # Full academic report, methodology, and graphs
├── Build_and_Train_Model.ipynb       # Model definition, training, and evaluation
├── data_loader.py                    # Custom dataset loading and transformation logic
├── requirements.txt                  # Python dependencies
└── videos/                           # Target directory for the dataset
    ├── Real_videos/
    └── Deepfakes_videos/