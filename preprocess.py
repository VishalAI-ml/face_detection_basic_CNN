import tensorflow as tf

# preprocess the dataset
def preprocess_lfw(label, image):
    # If image is 2D (grayscale), add channel dimension
    if len(image.shape) == 2:
        image = tf.expand_dims(image, axis=-1) 
    # If image is not RGB, convert to RGB    
    if image.shape[-1] != 3:
        image = tf.image.grayscale_to_rgb(image)
    # Resize the image to a fixed size
    image = tf.image.resize(image, [128, 128])
    
    # Normalize the pixel values to [0, 1]
    image = tf.cast(image, tf.float32) / 255.0
    
    return image, 1


# preprocess the images loaded from directory
def process_dir_images(image, label):
    # If image is 2D (grayscale), add channel dimension
    if len(image.shape) == 2:
        image = tf.expand_dims(image, axis=-1) 
    # If image is not RGB, convert to RGB    
    if image.shape[-1] != 3:
        image = tf.image.grayscale_to_rgb(image)
    # Resize the image to a fixed size
    image = tf.image.resize(image, [128, 128])
    
    # Normalize the pixel values to [0, 1]
    image = tf.cast(image, tf.float32) / 255.0
    
    return image, label