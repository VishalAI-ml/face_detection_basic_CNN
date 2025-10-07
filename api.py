from fastapi import FastAPI, File, UploadFile
from PIL import Image
import numpy as np
#import io
import tensorflow as tf

from preprocess import process_dir_images
from predict import predict_on_dataset, predict_face_API

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # read uploaded image bytes
    byte_image = await file.read()
    image = tf.image.decode_image(byte_image, channels=3)  # Decode image bytes to tensor
    image = tf.image.resize(image, [128, 128])  # Resize to model input size
    image = tf.expand_dims(image, axis = 0)  # Add batch dimension
    image = tf.cast(image, tf.float32)/255.0 # convert to float and normalize
    prediction = predict_face_API(image)
    if prediction > 0.5:
        result = f"The image is a face. Probability: {prediction:.2f}"
    else:
        result = f"The image is not a face. Probability: {prediction:.2f}"
    print(result)
    return {"result": result}