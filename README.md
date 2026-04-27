# CIFAR-10 Image Classifier

A Convolutional Neural Network (CNN) trained on the CIFAR-10 dataset to classify images into 10 categories: **airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck**.

Built with TensorFlow/Keras and deployable via a REST API.

---

## Table of Contents

- [Setup Instructions](#setup-instructions)
- [How to Train](#how-to-train)
- [How to Run the API](#how-to-run-the-api)
- [Sample Request & Response](#sample-request--response)
- [Model Performance](#model-performance)

---

## Setup Instructions

### Prerequisites

- Python 3.8+
- pip
- (Optional) NVIDIA GPU with CUDA, Apple Silicon with MPS, or Intel GPU with XPU — the code auto-detects the best available device

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/cifar10-classifier.git
cd cifar10-classifier
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**`requirements.txt`**

```
tensorflow>=2.12.0
tensorflow-datasets
torch
torchvision
numpy
pandas
scikit-learn
matplotlib
seaborn
opencv-python
Pillow
flask            # for the REST API
```

---

## How to Train

### Option A — Jupyter Notebook (Recommended for Experimentation)

Open and run all cells in the notebook:

```bash
jupyter notebook 1__1_.ipynb
```

The notebook will:
1. Download CIFAR-10 automatically via `tensorflow_datasets`
2. Apply preprocessing (normalization to `[0, 1]`) and augmentation (random crops, horizontal flips)
3. Build the CNN model
4. Train for **50 epochs** with Adam (`lr=1e-4`) and `sparse_categorical_crossentropy` loss
5. Evaluate and display a classification report + confusion matrix
6. Save the trained model to `model.keras`

### Option B — Script

```bash
python train.py
```

### Training Configuration

| Parameter       | Value                          |
|-----------------|--------------------------------|
| Optimizer       | Adam                           |
| Learning Rate   | 1e-4                           |
| Loss Function   | Sparse Categorical Crossentropy|
| Batch Size      | 32                             |
| Epochs          | 50                             |
| Input Shape     | (32, 32, 3)                    |
| Output Classes  | 10                             |

### Model Architecture

```
Input (32×32×3)
  │
  ├─ Conv2D(32, 3×3, relu) → BatchNorm → Conv2D(32, 3×3, relu) → MaxPool → Dropout(0.25)
  │
  ├─ Conv2D(64, 3×3, relu) → BatchNorm → Conv2D(64, 3×3, relu) → MaxPool → Dropout(0.25)
  │
  ├─ Flatten
  │
  ├─ Dense(128, relu) → BatchNorm → Dropout(0.5)
  │
  └─ Dense(10, softmax)  →  Predicted Class
```

### Device Selection

The code automatically selects the best available hardware:

```
MPS (Apple Silicon)  →  CUDA (NVIDIA GPU)  →  XPU (Intel)  →  CPU
```

---

## How to Run the API

Create a file named `app.py` in the project root:

```python
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf
import io

app = Flask(__name__)

CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]

model = tf.keras.models.load_model("model.keras")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    image = Image.open(io.BytesIO(file.read())).convert("RGB")

    img_resized = image.resize((32, 32))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array, verbose=0)
    class_idx = int(np.argmax(preds[0]))
    confidence = float(np.max(preds[0]))

    return jsonify({
        "predicted_class": CLASS_NAMES[class_idx],
        "confidence": round(confidence * 100, 2),
        "all_probabilities": {
            cls: round(float(prob) * 100, 2)
            for cls, prob in zip(CLASS_NAMES, preds[0])
        }
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
```

### Start the Server

```bash
python app.py
```

The API will be available at `http://localhost:5000`.

---

## Sample Request & Response

### Request

Send a `POST` request to `/predict` with an image file attached:

**Using `curl`:**

```bash
curl -X POST http://localhost:5000/predict \
     -F "image=@/path/to/your/image.jpg"
```

**Using Python `requests`:**

```python
import requests

with open("cat.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:5000/predict",
        files={"image": f}
    )

print(response.json())
```

### Response

```json
{
  "predicted_class": "cat",
  "confidence": 87.43,
  "all_probabilities": {
    "airplane":    0.12,
    "automobile":  0.08,
    "bird":        1.35,
    "cat":        87.43,
    "deer":        0.54,
    "dog":         9.21,
    "frog":        0.44,
    "horse":       0.38,
    "ship":        0.27,
    "truck":       0.18
  }
}
```

### Error Response

```json
{
  "error": "No image provided"
}
```

---

## Model Performance

After training for 50 epochs on CIFAR-10 (50,000 training / 10,000 test images):

### Overall Metrics

| Metric              | Score  |
|---------------------|--------|
| Test Accuracy       | ~75–78% |
| Macro Avg F1-Score  | ~0.75  |

> Exact values depend on hardware, random seed, and number of completed epochs.

### Per-Class Performance (Approximate)

| Class       | Precision | Recall | F1-Score |
|-------------|-----------|--------|----------|
| airplane    | 0.79      | 0.80   | 0.80     |
| automobile  | 0.87      | 0.86   | 0.87     |
| bird        | 0.68      | 0.63   | 0.65     |
| cat         | 0.59      | 0.57   | 0.58     |
| deer        | 0.74      | 0.75   | 0.75     |
| dog         | 0.65      | 0.67   | 0.66     |
| frog        | 0.80      | 0.84   | 0.82     |
| horse       | 0.81      | 0.82   | 0.82     |
| ship        | 0.85      | 0.87   | 0.86     |
| truck       | 0.85      | 0.84   | 0.85     |

> The model struggles most with **cat** and **dog** — a well-known challenge on CIFAR-10 due to high visual similarity between the two classes.

### Confusion Matrix

The confusion matrix is automatically plotted at the end of notebook training using `seaborn`. The most common misclassifications occur between:

- `cat` ↔ `dog`
- `bird` ↔ `airplane`
- `deer` ↔ `horse`

### Improving Performance

To push accuracy above 80%, consider:

- Using a **ResNet-18** or **EfficientNet** backbone via `torchvision.models`
- Adding stronger augmentation (e.g., `CutMix`, `MixUp`)
- Increasing epochs with a **learning rate scheduler** (e.g., cosine annealing)
- Using **transfer learning** from ImageNet pretrained weights

---

## Project Structure

```
cifar10-classifier/
├── 1__1_.ipynb        # Training notebook
├── app.py             # Flask inference API
├── train.py           # (Optional) standalone training script
├── model.keras        # Saved model (generated after training)
├── requirements.txt
└── README.md
```

---
