from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import numpy as np
import tensorflow as tf
import io

app = FastAPI()

CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer","dog", "frog", "horse", "ship", "truck"]

# Load model once (startup)
try:
    model = tf.keras.models.load_model("C:\\Users\\hari.s.kumar\\Desktop\\Harsh - AIML\\cifar10_simple_cnn.h5")
except Exception as e:
    raise RuntimeError(f"Model loading failed: {e}")


def preprocess_image(image: Image.Image):
    image = image.resize((32, 32))  # CIFAR-10 size
    image = np.array(image)

    if image.shape[-1] != 3:
        raise ValueError("Image must be RGB")

    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # ✅ Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    try:
        processed = preprocess_image(image)
        preds = model.predict(processed)

        class_idx = int(np.argmax(preds[0]))
        confidence = float(np.max(preds[0]))

        return {
            "class": CLASS_NAMES[class_idx],
            "confidence": round(confidence, 4)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))