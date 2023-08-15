import json
import numpy as np
from tensorflow import keras
import os

# Load the saved model
saved_model_path = "../mood_classification/results/814_PCRNN_2D_valence_50each/saved_model"
test_data_path = "test_audio_data_arousal.json"

def load_testing_data(test_data_path):
    with open(test_data_path, "r") as fp:
        test_data = json.load(fp)

    X_test = np.array(test_data["mfcc"])  # Adjust the key as per your data format
    y_test = np.array(test_data["labels"])  # Adjust the key as per your data format

    return X_test, y_test

X_test, y_test = load_testing_data(test_data_path)
X_test = X_test[..., np.newaxis]  # If needed, reshape your data for the model input

loaded_model = keras.models.load_model(saved_model_path)

# Make predictions
predictions = loaded_model.predict(X_test)

# If you have a classification task, you can get the predicted class indices:
predicted_class_indices = np.argmax(predictions, axis=1)

# Define your label list mapping class indices to labels
label_list = {
    0: "Valence_2",
    1: "Valence_3",
    2: "Valence_4",
    3: "Valence_5",
    4: "Valence_6",
    5: "Valence_7",
    6: "Valence_8"
}

# Assuming label_list contains the mapping of class indices to labels
predicted_labels = [label_list[index] for index in predicted_class_indices]
for i, label in enumerate(predicted_labels):
    print(f"Sample {i + 1}: Predicted Label: {label}")

print(predicted_labels)