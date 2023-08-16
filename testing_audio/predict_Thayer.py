import json
import numpy as np
from tensorflow import keras
import os

# Load the saved model
arousal_model_path = "../mood_classification/results/815_PCRNN_2D_arousal_50each/saved_model"
valence_model_path = "../mood_classification/results/814_PCRNN_2D_valence_50each/saved_model"
test_data_path = "test_hip_hop.json"

def load_testing_data(test_data_path):
    with open(test_data_path, "r") as fp:
        test_data = json.load(fp)

    X_test = np.array(test_data["mfcc"])  # Adjust the key as per your data format
    y_test = np.array(test_data["labels"])  # Adjust the key as per your data format

    return X_test, y_test

def scaling(num, in_min, in_max, out_min, out_max):
    return ((num - in_min) * (out_max - out_min)) / (in_max - in_min) + out_min

X_test, y_test = load_testing_data(test_data_path)
X_test = X_test[..., np.newaxis]  # If needed, reshape your data for the model input

loaded_arousal_model = keras.models.load_model(arousal_model_path)
loaded_valence_model = keras.models.load_model(valence_model_path)

# Make predictions
a_predictions = loaded_arousal_model.predict(X_test)
v_predictions = loaded_valence_model.predict(X_test)

# If you have a classification task, you can get the predicted class indices:
a_predicted_class_indices = np.argmax(a_predictions, axis=1)
v_predicted_class_indices = np.argmax(v_predictions, axis=1)

# Define your label list mapping class indices to labels
a_label_list = {
    0: "Arousal_2",
    1: "Arousal_3",
    2: "Arousal_4",
    3: "Arousal_5",
    4: "Arousal_6",
    5: "Arousal_7"
}

v_label_list = {
    0: "Valence_2",
    1: "Valence_3",
    2: "Valence_4",
    3: "Valence_5",
    4: "Valence_6",
    5: "Valence_7",
    6: "Valence_8"
}

# Assuming label_list contains the mapping of class indices to labels
a_indices = []
v_indices = []

a_predicted_labels = [a_label_list[index] for index in a_predicted_class_indices]
for i, label in enumerate(a_predicted_labels):
    print(f"Sample {i + 1}: Arousal Predicted Label: {label}")
    a_indices.append(int(label.split("_")[-1]))

v_predicted_labels = [v_label_list[index] for index in v_predicted_class_indices]
for i, label in enumerate(v_predicted_labels):
    print(f"Sample {i + 1}: Valence Predicted Label: {label}")
    v_indices.append(int(label.split("_")[-1]))

print("Arousal Mean : ", np.mean(a_indices), "Arousal Std : ", np.std(a_indices))
print("Valence Mean : ", np.mean(v_indices), "Valence Std : ", np.std(v_indices))

scaled_Arousal_mean = scaling(np.mean(a_indices), 1, 9, 0, 1)
scaled_Valence_mean = scaling(np.mean(v_indices), 1, 9, 0, 1)

print("Scaled Arousal Mean : ", scaled_Arousal_mean)
print("Scaled Valence Mean : ", scaled_Valence_mean)


# 0 ~ 325 


if scaled_Valence_mean < (1 / 3) : 
    if scaled_Arousal_mean < 0.25 :
        mood = "Sad"
    elif scaled_Arousal_mean < 0.5 :
        mood = "Bored"
    elif scaled_Arousal_mean < 0.75 :
        mood = "Nervous"
    else:
        mood = "Angry"
elif scaled_Valence_mean < (2/3) :
    if scaled_Arousal_mean < 0.25 :
        mood = "Sleepy"
    elif scaled_Arousal_mean < 0.75 :
        mood = "Calm"
    else :
        mood = "Excited"
else :
    if scaled_Arousal_mean < 0.25 :
        mood = "Peaceful"
    elif scaled_Arousal_mean < 0.5 :
        mood = "Relaxed"
    elif scaled_Arousal_mean < 0.75 :
        mood = "Pleased"
    else:
        mood = "Happy"

print("Mood of this song : ", mood)