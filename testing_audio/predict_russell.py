from tensorflow import keras
import numpy as np

# Load the saved model
saved_model_path = "../mood_classification/results/814_PCRNN_2D_valence_50each/saved_model"
loaded_model = keras.models.load_model(saved_model_path)
X_test_path = ""

# Assuming X_test is your testing data
# Make sure X_test is in the same format (shape) as the input of the loaded model
predictions = loaded_model.predict(X_test)

# If you have a classification task, you can get the predicted class indices:
predicted_class_indices = np.argmax(predictions, axis=1)