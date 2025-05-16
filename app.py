import numpy as np
import tensorflow as tf
from keras.preprocessing.image import img_to_array
from mtcnn import MTCNN
import cv2
import logging
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Load model
logging.info("Loading model...")
model = tf.keras.models.load_model("confidence_measuring_ver4.keras")
logging.info("Model loaded successfully.")

# Init FastAPI app
app = FastAPI()

# Allow CORS for frontend use
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

picture_size = 48

def detect_and_crop_face(image: np.ndarray):
    try:
        logging.info("Running face detection...")
        img_rgb = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        detector = MTCNN()
        faces = detector.detect_faces(img_rgb)
        if not faces:
            return None, "No face detected"

        x, y, w, h = faces[0]['box']
        x, y = max(0, x), max(0, y)
        face = image[y:y+h, x:x+w]
        gray_face = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
        resized_face = cv2.resize(gray_face, (picture_size, picture_size))
        return resized_face, None
    except Exception as e:
        return None, f"Detection error: {str(e)}"

def predict_confidence(image: np.ndarray):
    cropped_face, error = detect_and_crop_face(image)
    if error:
        return error

    img_array = img_to_array(cropped_face) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    img_array = np.expand_dims(img_array, axis=-1)

    prediction = model.predict(img_array)
    confidence_score = (1 - prediction[0][0]) * 100
    predicted_class = "Confident" if prediction[0][0] < 0.5 else "Not Confident"
    result = f"{predicted_class} (Confidence: {confidence_score:.2f}%)"
    return result

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        np_image = np.array(image)
        result = predict_confidence(np_image)
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}
