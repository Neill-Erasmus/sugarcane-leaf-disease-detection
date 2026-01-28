# Sugarcane Leaf Disease Detection

## Project Overview
This project implements a computer vision system for detecting diseases in sugarcane leaves.

**Motivation:** Sugarcane is a critical agricultural crop, and early detection of leaf diseases can significantly improve yield and reduce crop loss. This project demonstrates how deep learning and computer vision can be applied to agricultural disease diagnosis, bridging real-world impact with modern ML engineering practices.

**Objective:** classify sugarcane leaves into five categories: Healthy, Mosaic, RedRot, Rust, Yellow

## Folder Structure

```
sugarcane-leaf-disease-detection/
├── api/                 #FastAPI
├── data/                #train / val / test image folders
├── experiments/         #checkpoints, metrics,confusion matrices
├── src/                 #training, evaluation, models, data loaders
├── Dockerfile
├── requirements.txt
├── config.yaml
└── README.md
```

## Tech Stack

- Python 3.10
- PyTorch & Torchvision
- FastAPI
- Docker
- NumPy, scikit-learn

## Dataset
- The dataset consists of images organized into `train/`, `val/`, and `test/` folders.  
- Data splits:
  - Train: 80%
  - Validation: 20% (from original train set)
  - Test: separate untouched set

## Setup / Installation

```
git clone https://github.com/yourusername/sugarcane-leaf-disease-detection.git
cd sugarcane-leaf-disease-detection

#create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

pip install --upgrade pip
pip install -r requirements.txt
```

## Running the API

The API loads the best-performing trained model from the experiments/ directory as specified in config.yaml.

```
uvicorn api.app:app --host 0.0.0.0 --port 8000
```
- **Endpoint:** /predict (POST)
- **Request:** form-data with file field containing an image (jpg or png)
- **Response Example:**
```
{
  "predicted_class": "RedRot",
  "probabilities": {
    "Healthy": 0.01,
    "Mosaic": 0.02,
    "RedRot": 0.93,
    "Rust": 0.03,
    "Yellow": 0.01
  }
}
```

## Running with Docker
**Build and run the container:**

```
docker build -t sugarcane-api .

docker run -d -p 8000:8000 --name sugarcane-api-container sugarcane-api
```

**Access API at:** ```http://localhost:8000/docs#/default/predict_disease_predict_post```

**Stop and remove container when done:**
```
docker stop sugarcane-api-container

docker rm sugarcane-api-container
```

## Training and Evaluation

```
#train models
python src/training/train.py                  
python src/training/train_resnet.py           
python src/training/train_resnet_finetuned.py 

#evaluate models
python src/evaluation/evaluate.py
python src/evaluation/evaluate_resnet.py
python src/evaluation/evaluate_resnet_finetuned.py
```

- Trained models and metrics are stored in experiments/ folder
- Use config.yaml to adjust hyperparameters, batch size, epochs, and paths

## Results Summary

|Model|Accuracy|Macro F1|
|---|---|---|
|Baseline CNN|0.75|0.74|
|ResNet-50 (Frozen)|0.79|0.78|
|ResNet-50 (Fine-Tuned)|0.98|0.98|

- Fine-tuned ResNet-50 resolves Healthy/Mosaic confusion and achieves near-perfect classification.

## Key Observations

- Transfer learning boosts performance significantly
- Freezing backbone allows fast convergence but limits domain adaptation
- Fine-tuning deeper layers yields best results

## Configuration

All key parameters are defined in `config.yaml`, including:

- Training hyperparameters (batch size, learning rate, epochs)
- Dataset paths
- Model checkpoint paths
- API input constraints

This allows easy experimentation without modifying code.