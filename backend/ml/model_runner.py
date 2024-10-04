from keras.api.preprocessing import image
import numpy as np

def run_model(model, img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    predictions = model.predict(img_array)

    class_labels = ['COVID', 'NORMAL', 'PNEUMONIA', 'TUBERCULOSIS']
    predicted_class = np.argmax(predictions, axis=1)
    predicted_class_name = class_labels[predicted_class[0]]

    return predicted_class_name
