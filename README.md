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
  - Exact image counts have been omitted but class balance is approximately uniform.
- Data splits:
  - Train: 80%
  - Validation: 20% (from original train set)
  - Test: separate untouched set

**Note:** Images are resized to 256x256 pixels and augmented during training.

---

## Folder Structure

```
sugarcane-leaf-disease-detection/  
├── data/ # train, val, test images  
├── src/  
│ ├── data/ # Data loaders  
│ ├── models/ # Baseline CNN, ResNet (Frozen and Fine-Tuned) 
│ ├── training/ # Training script  
│ └── evaluation/ # Evaluation script  
├── experiments/ # Checkpoints, logs  
├── notebooks/ # Optional EDA or visualization  
├── api/ # FastAPI endpoint   
├── docker/ # Dockerfile  
├── requirements.txt  
├── config.yaml # Hyperparameters and paths  
└── README.md
```
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

```
experiments/baseline_cnn/checkpoints/baseline_cnn_best.pth
```

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

---

## Transfer Learning: ResNet-50 (Frozen Backbone)

After establishing a baseline using a custom CNN, a transfer learning approach was implemented using a pretrained **ResNet-50** model. The motivation was to leverage strong general-purpose visual features learned from ImageNet and adapt them to the sugarcane leaf disease classification task.

---

### Model Architecture

---

- Backbone: **ResNet-50 pretrained on ImageNet**
- Modification:
  - Final fully connected layer replaced with a linear layer matching the number of disease classes (5)
- Training strategy:
  - **All ResNet backbone layers frozen**
  - Only the final classification head trained

This setup allows fast convergence while evaluating how well pretrained features generalize to agricultural imagery without domain-specific fine-tuning.

---

### Training Configuration

---

- Input size: `224 × 224`
- Optimizer: Adam
- Loss function: CrossEntropyLoss
- Learning rate: `1e-3`
- Epochs: `15`
- Data augmentation:
  - Random horizontal flip
  - Random rotation
  - Normalization using ImageNet statistics

Training was executed using:

```bash
python -m src.training.train_resnet
```

The best-performing model checkpoint was saved automatically during training.

---

## Training Performance

| Epoch | Validation Accuracy |
|---|---|
| 1 | 0.476 |
| 3 | 0.695|
|5| 0.730|
|8|0.752|
|11|0.777|
|15|0.749|

The model converged rapidly, indicating effective reuse of pretrained features.

---

## Evaluation Results
Evaluation was performed on a held-out test set using:
```
python -m src.evaluation.evaluate_resnet
```

## Classification Report

|Class|Precision|Recall|F1-Score|
|---|---|---|---|
|Healthy|1.00|0.41|0.59|
|Mosaic|0.63|0.97|0.76|
|RedRot|0.73|0.87|0.80|
|Rust|0.93|0.91|0.92|
|Yellow|0.85|0.79|0.82|

- Overall accuracy: 0.79
- Macro F1-score: 0.78
- Weighted F1-score: 0.78

---

## Confusion Matrix
```
[[43 41 17  1  2]
 [ 0 89  0  0  3]
 [ 0  0 90  5  8]
 [ 0  4  4 93  1]
 [ 0  8 12  1 80]]
```

---

## Comparison to Baseline CNN

|Model|Accuracy|Macro F1|
|---|---|---|
|Baseline CNN|0.75|0.74|
|ResNet-50 (Frozen)|0.79|0.78|

The frozen ResNet-50 model outperforms the baseline CNN, particularly on Rust, RedRot, and Yellow disease classes.

However, a notable weakness is observed in the Healthy class, where recall is low. This suggests that while pretrained ImageNet features are effective, they are not fully adapted to domain-specific visual patterns in sugarcane leaves.

---

## Key Observations

- Transfer learning provides a clear performance gain over a custom CNN
- Frozen backbones converge quickly but may struggle with subtle domain-specific distinctions
- Significant confusion remains between Healthy and Mosaic leaves

---

## Next Steps

- Fine-tune the deeper layers of ResNet-50 (starting with layer4)
- Reduce learning rate for stable fine-tuning
- Evaluate whether domain adaptation improves Healthy-class recall and overall performance

---

## Transfer Learning: ResNet-50 (Fine-Tuned)

To address the limitations observed with the frozen ResNet-50 model—particularly poor recall for the Healthy class—a fine-tuning strategy was implemented.

Rather than training only the classification head, the deepest convolutional layers of the network were unfrozen to allow domain-specific feature adaptation.

## Fine Tuning Strategy

- Base model: ResNet-50 pretrained on ImageNet
- Layers frozen:
  - All layers except layer4
- Layers trained:
  - layer4
  - Final fully connected classification layer
- Motivation:
  - Preserve general low-level visual features
  - Adapt high-level semantic features to sugarcane leaf textures and disease patterns

## Training Configuration (Fine-Tuned)

- Input size: 224 × 224
- Optimizer: Adam
- Loss function: CrossEntropyLoss
- Learning rate: 1e-4
- Epochs: 10
- Batch size: 32
- Data augmentation:
  - Random horizontal flip
  - Random rotation
  - Color jitter
  - ImageNet normalization

Training command:
```
python -m src.training.train_resnet_finetuned
```
The best-performing model was automatically saved to:
```
experiments/resnet_finetuned/checkpoints/resnet50_finetuned_best.pth
```

## Fine-Tuned Training Performance

|Epoch|Validation Accuracy|
|---|---|
|1|0.873|
|2|0.913|
|3|0.906|
|6|0.916|
|9|0.921|
|10|0.931|

Fine-tuning resulted in rapid convergence and strong generalization, with minimal overfitting.

## Fine-Tuned Evaluation Results

Evaluation was performed on the held-out test set using:

```
python -m src.evaluation.evaluate_resnet_finetuned
```

## Classification Report

|Class|Precision|Recall|F1-Score|
|---|---|---|---|
|Healthy|0.97|0.99|0.98|
|Mosaic|0.98|0.98|0.98|
|RedRot|0.96|0.97|0.97|
|Rust|0.98|0.96|0.97|
|Yellow|0.99|0.98|0.99|

- Overall Accuracy: 0.98
- Macro F1-Score: 0.98
- Weighted F1-Score: 0.98

## Confusion Matrix

```
[[103   1   0   0   0]
 [  2  90   0   0   0]
 [  0   0 100   2   1]
 [  1   0   3  98   0]
 [  0   1   1   0  99]]
```

The fine-tuned model shows near-perfect separation between all disease classes and resolves the Healthy/Mosaic confusion observed in earlier models.

## Model Comparison Summary

|Model|Accuracy|Macro F1|
|---|---|---|
|Baseline CNN|0.75|0.74|
|ResNet-50 (Frozen)|0.79|0.78|
|ResNet-50 (Fine-Tuned)|0.98|0.98|

## Key Findings

- Transfer learning significantly improves performance over a custom CNN
- Freezing the backbone provides quick gains but limits domain adaptation
- Fine-tuning higher-level layers yields dramatic performance improvements
- Agricultural imagery benefits strongly from domain-specific feature refinement

## Experiments & Reproducibility

All experiments are stored in a structured and reproducible format:

```
experiments/
├── baseline_cnn/
│   ├── checkpoints/
│   ├── confusion_matrix.txt
│   ├── metrics.json
│   └── notes.md
├── resnet_frozen/
│   ├── checkpoints/
│   ├── confusion_matrix.txt
│   ├── metrics.json
│   └── notes.md
└── resnet_finetuned/
    ├── checkpoints/
    ├── confusion_matrix.txt
    ├── metrics.json
    └── notes.md
```

Each experiment folder contains:
- Best model checkpoint
- Full evaluation metrics in machine-readable format
- Confusion matrix
- Experiment notes and observations

## Inference

