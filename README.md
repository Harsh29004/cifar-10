#  CIFAR-10 Image Classification API (FastAPI + TensorFlow)

##  Overview

This project is a production-style image classification system built using **TensorFlow/Keras** and deployed via **FastAPI**.
It classifies images into 10 categories from the CIFAR-10 dataset.

**Classes:**

```
airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
```

---

#  Setup Instructions

## 1. Clone the project

```bash
git clone <your-repo-url>
cd <project-folder>
```

## 2. Create virtual environment (recommended)

```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Linux/Mac
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Required Libraries (if no requirements.txt)

```bash
pip install tensorflow fastapi uvicorn pillow numpy
```

---

#  Model Training

## Run training script

(Assuming you trained in Colab / notebook)

```bash
python train.py
```

OR from Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```
1.ipynb
```

## Key Training Steps

* Load CIFAR-10 dataset
* Normalize images
* Build CNN model
* Train model
* Save model

## Model Saving

```python
model.save("model.h5")
```

---

#  Run API Server

## Start FastAPI server

```bash
uvicorn main:app --reload
```

## Server runs at:

```
http://127.0.0.1:8000
```

## Swagger UI:

```
http://127.0.0.1:8000/docs
```

---

#  API Usage

## Endpoint

```
POST /predict
```

## Request (Form-Data)

* Key: `file`
* Value: Image file (jpg/png)

---

##  Sample Request (Python)

```python
import requests

url = "http://127.0.0.1:8000/predict"

files = {"file": open("test.jpg", "rb")}
response = requests.post(url, files=files)

print(response.json())
```

---

##  Sample Response

```json
{
  "class": "cat",
  "confidence": 0.92
}
```

---

#  API Code Structure (Simplified)

```python
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))
    image = image.resize((32, 32))
    
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    class_index = np.argmax(predictions)

    return {
        "class": CLASS_NAMES[class_index],
        "confidence": float(np.max(predictions))
    }
```

---

#  Model Performance

| Metric       | Value (Approx) |
| ------------ | -------------- |
| Training Acc | ~80%        |
| testing Acc   | ~65%        |
| Loss         | Moderate       |

## Observations

* Performs well on clear images
* Struggles with:

  * low resolution inputs
  * noisy backgrounds
* Overfitting controlled using:

  * normalization
  * dropout (if used)

---

#  Limitations (Don’t Ignore This)

* CIFAR-10 images are **32x32** → real-world images lose detail
* Model is not robust for production-scale deployment
* No input validation / security hardening yet
* No logging or monitoring

---

# Improvements (If You’re Serious)

If you actually want this to stand out in interviews:

1. Use **ResNet / EfficientNet**
2. Add **data augmentation**
3. Deploy on **Docker**
4. Add **batch prediction endpoint**
5. Integrate **real-time logging**
6. Add **confidence threshold filtering**

---

# Project Structure

```
project/
│
├── main.py          # FastAPI app
├── model.h5         # Trained model
├── train.py         # Training script
├── requirements.txt
├── README.md
└── notebooks/
    └── 1.ipynb
```

---

