import json
import numpy as np
from tensorflow import keras
import os

# Load the saved model
arousal_model_path = "../mood_classification/results/812_PCRNN_2D_arousal_50each/saved_model"
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

# add mood_extractor

def mood_extractor(valence, arousal):
    if valence <= 0.4 and arousal <= 0.4:
        mood = "Depressed"
    elif valence <= 0.4 and arousal > 0.4 and arousal <= 0.6:
        mood = "Angry"
    elif valence <= 0.4 and arousal > 0.6:
        mood = "Distressed"
    elif valence > 0.4 and valence <= 0.6 and arousal <= 0.4:
        mood = "Bored"
    elif valence > 0.4 and valence <= 0.6 and arousal > 0.4 and arousal <= 0.6:
        mood = "Calm"
    elif valence > 0.4 and valence <= 0.6 and arousal > 0.6:
        mood = "Content"
    elif valence > 0.6 and arousal <= 0.4:
        mood = "Sad"
    elif valence > 0.6 and arousal > 0.4 and arousal <= 0.6:
        mood = "Relaxed"
    else:
        mood = "Happy"
    
    return mood

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

a_indices = []
v_indices = []
a_dict = {}
v_dict = {}
mood = []

####Recreate Main to give average scores and output mood####
#probably need to use a circumplex equation/model to average out values

# Assuming label_list contains the mapping of class indices to labels

a_predicted_labels = [a_label_list[index] for index in a_predicted_class_indices]
for i, label in enumerate(a_predicted_labels):
    print(f"Sample {i + 1}: Arousal Predicted Label: {label}")
    arousal_val = int(label.split("_")[-1])
    if arousal_val not in a_dict:
        a_dict[arousal_val] = 0
    else:
        a_dict[arousal_val] += 1
    a_indices.append(arousal_val)

v_predicted_labels = [v_label_list[index] for index in v_predicted_class_indices]
for i, label in enumerate(v_predicted_labels):
    print(f"Sample {i + 1}: Valence Predicted Label: {label}")
    valence_val = int(label.split("_")[-1])
    if valence_val not in v_dict:
        v_dict[valence_val] = 0
    else:
        v_dict[valence_val] += 1
    v_indices.append(valence_val)

print("Arousal Mean : ", np.mean(a_indices), "Arousal Std : ", np.std(a_indices))
print("Valence Mean : ", np.mean(v_indices), "Valence Std : ", np.std(v_indices))

scaled_Arousal_mean = scaling(np.mean(a_indices), 1, 9, 0, 1)
scaled_Valence_mean = scaling(np.mean(v_indices), 1, 9, 0, 1)

print("Scaled Arousal Mean : ", scaled_Arousal_mean)
print("Scaled Valence Mean : ", scaled_Valence_mean)

# mood extraction based on mean value
mood.append(mood_extractor(scaled_Valence_mean, scaled_Arousal_mean))

print(a_dict)
print(v_dict)
# mood extraction based on the most frequent value
a_mod = []
v_mod = []

a_max =  max(a_dict, key=a_dict.get)
a_mod.append(a_max)

v_max =  max(v_dict, key=v_dict.get)
v_mod.append(v_max)

scaled_Arousal = scaling(np.mean(a_mod), 1, 9, 0, 1)
scaled_Valence = scaling(np.mean(v_mod), 1, 9, 0, 1)

print("Scaled Most Frequent Arousal : ", scaled_Arousal)
print("Scaled Most Frequent Valence : ", scaled_Valence)

mood.append(mood_extractor(scaled_Valence, scaled_Arousal))

print("Mood of this song : ", mood)