# Sugarcane Leaf Disease Detection

![Python](https://img.shields.io/badge/python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)

A compact toolkit for detecting diseases in sugarcane leaves using deep learning (classification into Healthy, Mosaic, RedRot, Rust, Yellow).

## Table of Contents

- [Project Overview](#project-overview)
- [Folder Structure](#folder-structure)
- [Tech Stack](#tech-stack)
- [Dataset](#dataset)
- [Setup / Installation](#setup--installation)
- [Quick Start](#quick-start)
- [Running the API](#running-the-api)
- [Running with Docker](#running-with-docker)
- [Training and Evaluation](#training-and-evaluation)
- [Results Summary](#results-summary)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

## Project Overview

This repository implements a computer-vision pipeline and API for sugarcane leaf disease classification. It demonstrates training, evaluation, and deployment patterns for image classification using PyTorch and FastAPI.

## Folder Structure

```
sugarcane-leaf-disease-detection/
├── api/                 # FastAPI application and model serving
├── data/                # train / val / test image folders
├── experiments/         # checkpoints, metrics, confusion matrices
├── src/                 # training, evaluation, models, data loaders
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

- Images are organized into `train/`, `val/`, and `test/` folders under `data/`.
- Typical split: Train ~80%, Validation ~20% (from training set), Test = held-out set.

## Setup / Installation

Clone and install dependencies:

```bash
git clone https://github.com/Neill-Erasmus/sugarcane-leaf-disease-detection.git
cd sugarcane-leaf-disease-detection

python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate    # Windows

pip install --upgrade pip
pip install -r requirements.txt
```

## Quick Start

Run the API locally (serves the model from `experiments/` as configured in `config.yaml`):

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

Test prediction with curl:

```bash
curl -X POST "http://localhost:8000/predict" -F "file=@/path/to/image.jpg"
```

## Running the API

The API exposes `/predict` (POST) which accepts an image file and returns the predicted class and class probabilities.

Example response:

```json
{
  "predicted_class": "RedRot",
  "probabilities": {"Healthy": 0.01, "Mosaic": 0.02, "RedRot": 0.93, "Rust": 0.03, "Yellow": 0.01}
}
```

## Running with Docker

Build and run:

```bash
docker build -t sugarcane-api .
docker run -d -p 8000:8000 --name sugarcane-api-container sugarcane-api
```

Stop and remove container:

```bash
docker stop sugarcane-api-container && docker rm sugarcane-api-container
```

## Training and Evaluation

Train and evaluate using provided scripts:

```bash
# Train
python src/training/train.py
python src/training/train_resnet.py
python src/training/train_resnet_finetuned.py

# Evaluate
python src/evaluation/evaluate.py
python src/evaluation/evaluate_resnet.py
python src/evaluation/evaluate_resnet_finetuned.py
```

Trained artifacts and metrics are stored under `experiments/`.

## Results Summary

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| Baseline CNN | 0.75 | 0.74 |
| ResNet-50 (Frozen) | 0.79 | 0.78 |
| ResNet-50 (Fine-Tuned) | 0.98 | 0.98 |

## Configuration

All key parameters (hyperparameters, paths, checkpoint locations) are configured in `config.yaml`.

## Contributing

See [CONTRIBUTING.md](Contributing.md) for contribution guidelines, code style, and the development workflow. We welcome issues, pull requests, and documentation improvements.

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.