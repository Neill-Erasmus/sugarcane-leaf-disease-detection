# Sugarcane Leaf Disease Detection

## Project Overview
This project implements a **production-ready computer vision system** for detecting diseases in sugarcane leaves.  
The goal is to classify leaf images into one of five categories: Healthy, Mosaic, RedRot, Rust, or Yellow.

This repository demonstrates:
- End-to-end deep learning workflow
- Baseline model development from scratch
- Training, validation, and evaluation best practices
- Reproducible and modular code structure

---

## Dataset
- The dataset consists of RGB images organized into `train/`, `val/`, and `test/` folders.  
- Class distribution:
  - Healthy: XXX images
  - Mosaic: XXX images
  - RedRot: XXX images
  - Rust: XXX images
  - Yellow: XXX images
- Data splits:
  - Train: 80%
  - Validation: 20% (from original train set)
  - Test: separate untouched set

**Note:** Images are resized to 256x256 pixels and augmented during training.

---

## Folder Structure

sugarcane-leaf-disease-detection/  
├── data/ # train, val, test images  
├── src/  
│ ├── data/ # Data loaders  
│ ├── models/ # Baseline CNN  
│ ├── training/ # Training script  
│ └── evaluation/ # Evaluation script  
├── experiments/ # Checkpoints, logs  
├── notebooks/ # Optional EDA or visualization  
├── api/ # FastAPI endpoint   
├── docker/ # Dockerfile  
├── requirements.txt  
├── config.yaml # Hyperparameters and paths  
└── README.md

---

## Baseline Model

- **Architecture:** Custom CNN with 3 convolutional layers, max-pooling, dropout, and 2 fully connected layers.
- **Input size:** 256x256 RGB images
- **Number of classes:** 5
- **Framework:** PyTorch
- **Loss:** CrossEntropyLoss
- **Optimizer:** Adam (lr=1e-3)
- **Batch size:** 32
- **Epochs:** 20
- **Data augmentation:** Random horizontal/vertical flips, rotation, brightness/contrast/saturation jitter

**Rationale:**  
The shallow CNN serves as a **baseline** to validate the data pipeline, training procedure, and initial model performance before moving to transfer learning.

---

## Training Procedure

1. Run the training script:

```bash
python -m src.training.train
```

2. The best model checkpoint is automatically saved to:

```experiments/checkpoints/baseline_cnn_best.pth```

3. experiments/checkpoints/baseline_cnn_best.pth

---

## Evaluation

1. Run the evaluation script:

```python -m src.evaluation.evaluate```

2. Generates:

- Classification report (precision, recall, F1-score per class)
- Confusion matrix

3. These metrics form the baseline performance for comparison with future model upgrades.

---

## Baseline Results

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
|Healthy|0.68|0.88|0.76|104|
|Mosaic|0.78|0.51|0.62|92|
|RedRot|0.83|0.58|0.69|103|
|Rust|0.63|0.81|0.71|102|
|Yellow|0.90|0.93|0.91|101|

**Overall Accuracy**: 0.75

**Confusion Matrix:**
```
[[91  2  4  7  0]
 [30 47  0 15  0]
 [ 3  4 60 26 10]
 [ 9  2  7 83  1]
 [ 1  5  1  0 94]]
 ```

 ---

## Notes

- Baseline CNN is intended for initial benchmarking.
- Next steps: upgrade to transfer learning with ResNet/EfficientNet for higher accuracy and real-world robustness.