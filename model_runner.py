from keras.api.models import load_model
from keras.api.preprocessing import image
import numpy as np

model = load_model('chest_diagnosis_model.keras')

img_path = 'test\PNEUMONIA\person1_virus_7.jpeg'
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

# Make a prediction
predictions = model.predict(img_array)

# # Map predicted class to labels
print(predictions)
class_labels = ['COVID', 'NORMAL', 'PNEUMONIA', 'TUBERCULOSIS']
predicted_class = np.argmax(predictions, axis=1)
predicted_class_name = class_labels[predicted_class[0]]

print(f"Predicted Class: {predicted_class_name}")
