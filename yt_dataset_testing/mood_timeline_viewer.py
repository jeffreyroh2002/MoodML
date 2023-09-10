import json
import numpy as np
from tensorflow import keras
import os

# Load the saved model
saved_model_path = "../mood_classification/results/909_PCRNN_2D_binary_valence_dataset_with_lyrics_3sec/saved_model"
test_data_path = "p4_8songs_3sec.json"
valence_mfcc_path = "../binary_classification/json_files/valence_dataset_lyrics.json"


def load_testing_data(test_data_path):
    with open(test_data_path, "r") as fp:
        test_data = json.load(fp)

    X_test = np.array(test_data["mfcc"])  # Adjust the key as per your data format
    y_test = np.array(test_data["labels"])  # Adjust the key as per your data format
    filenames = test_data["filenames"]

    return X_test, y_test, filenames

def load_mfcc_labels(model_saved_mfcc):
    with open(model_saved_mfcc, "r") as fp:
        data = json.load(fp)

    mfcc_labels = np.array(data["mapping"])  # Adjust the key as per your data format

    return mfcc_labels

X_test, y_test, filenames = load_testing_data(test_data_path)
X_test = X_test[..., np.newaxis]  # If needed, reshape your data for the model input

loaded_model = keras.models.load_model(saved_model_path)

# Make predictions
predictions = loaded_model.predict(X_test)

# If you have a classification task, you can get the predicted class indices:
predicted_class_indices = np.argmax(predictions, axis=1)

# Define label list using mapping class indices
v_mfcc_labels = load_mfcc_labels(valence_mfcc_path)

label_list = {}
for i in range (len(v_mfcc_labels)):
    label_list[i] = v_mfcc_labels[i]

print(set(filenames))

Song_list = set(filenames)
Song_list = list(Song_list)
print(Song_list)
Sorted_Song_list = sorted(Song_list)
Song_list = {label : [] for label in Sorted_Song_list}

# Assuming label_list contains the mapping of class indices to labels
counter = 0
time = 0
predicted_labels = [label_list[index] for index in predicted_class_indices]

for i, label in enumerate(predicted_labels):
    f_name = filenames[i]
    Song_list[f_name].append(i)


for f in Sorted_Song_list:
    predicted_idx = Song_list[f]
    time = 0

    for idx in predicted_idx:
        print(f"{3*time} sec Sample {f} : Predicted Label: {predicted_labels[idx]}")
        time += 1

print(Sorted_Song_list)
print(Song_list)

