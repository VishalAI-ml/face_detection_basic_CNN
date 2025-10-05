import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

# Load the model
model = load_model('E:/AI-ML-DL/saved models/face_detection_model.h5')   # path to your saved model

# function to process loaded single image and make prediction
def predict_on_image(image_path):
    img = load_img(image_path, target_size=(128,128))
    img = img_to_array(img, dtype='float32')/ 255.0
    img = np.expand_dims(img, axis=0) # add batch dimension because model expects a batch of images
    prediction = model.predict(img)
    print(f"Prediction shape: {prediction}")
    return prediction[0][0]   # return the predicted probability for the face presence, was shped like [[0.8]] which is 2d array



# function to predict on a batch of images from dataset and visualize the results
def predict_on_dataset(test_images, num_batches):
    for images, labe in test_images.take(num_batches):
        print(images.shape)
        for img, lb in zip(images, labe):
            pred_prob = model.predict(tf.expand_dims(img, axis=0))
            pred_label = 1 if pred_prob > 0.5 else 0
            if pred_label == 1 :
                plt.figure()
                plt.imshow(img.numpy())
                plt.title(f"Actual: {lb}, Pred: {pred_label}, Prob: {pred_prob[0][0]:.2f})")
                plt.axis('off')  # Hide axes
                plt.show()
    return 1