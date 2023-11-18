from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Load the trained model
model = load_model('/Users/mathaios/Desktop/SkinCancerClassification/SkinCancerClassification.keras')

lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}

# Load a single image for testing (replace 'path_to_your_image.jpg' with the actual path)
test_image_path = '/Users/mathaios/Desktop/SkinCancerClassification/Dermatofibroma.jpeg'
test_image = image.load_img(test_image_path, target_size=(75, 100))
test_image_array = image.img_to_array(test_image)
test_image_array = np.expand_dims(test_image_array, axis=0)
test_image_array = test_image_array / 255.0  # Normalize pixel values to be between 0 and 1

# Make predictions
predictions = model.predict(test_image_array)

# Get the predicted class
predicted_class_index = np.argmax(predictions)
predicted_class = list(lesion_type_dict.keys())[predicted_class_index]
predicted_class_name = lesion_type_dict[predicted_class]
possibility = predictions[0][predicted_class_index] * 100

if (possibility > 60):
    print("Predicted class:", predicted_class)
    print("Predicted class name:", predicted_class_name, ":", possibility,"%")
else:
    print("Predicted Class Name : Not Sure")



plt.imshow(test_image)
plt.title("Test Image")
plt.show()