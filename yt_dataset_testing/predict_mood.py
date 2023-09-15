import json
import numpy as np
from tensorflow import keras
import plotly.graph_objs as go
import os

# Load the saved model
saved_model_path = "../mood_classification/results/914_PCRNN_2D_lyrics_reduced_lr/saved_model"
test_data_path = "json_files/p4_8songs_3sec.json"
model_saved_mfcc = "../dataset_lyrics_extracted/background_dataset.json"
output_dir = "new_radar_results/bg_radar_915_4mood"  # Directory to save individual radar chart images

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

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
mfcc_labels = load_mfcc_labels(model_saved_mfcc)

loaded_model = keras.models.load_model(saved_model_path)

# Make predictions
predictions = loaded_model.predict(X_test)

# If you have a classification task, you can get the predicted class indices:
predicted_class_indices = np.argmax(predictions, axis=1)

# Define your label list mapping class indices to labels
label_list = {}
for i in range (len(mfcc_labels)):
    label_list[i] = mfcc_labels[i]

Song_list = set(filenames)
Song_list = list(Song_list)
Sorted_Song_list = sorted(Song_list)
Song_list = {label : [] for label in Sorted_Song_list}

# sort labels 
for i, label in enumerate(predicted_class_indices):
    f_name = filenames[i]
    Song_list[f_name].append(i)


# Initialize variables for percentage calculation
segment_count = 0
label_counts = {label: 0 for label in label_list.values()}
song_radar_values = np.zeros(len(label_list))
song_index = 0

for f in Sorted_Song_list:
    predicted_idx = Song_list[f]

    for idx in predicted_idx:
        label = label_list[predicted_class_indices[idx]]

        # Update label counts for percentage calculation
        label_counts[label] += 1
        segment_count += 1
        song_radar_values[predicted_class_indices[idx]] += 1

    # Calculate the average radar values for the song
    avg_radar_values = song_radar_values / segment_count
    # Output the averaged radar values to the terminal
    print(f"{f} Average Radar Values:")
    for label, avg_value in zip(label_list.values(), avg_radar_values):
        print(f"{label}: {avg_value:.2f}")

    radar_chart_trace = go.Scatterpolar(
            r=avg_radar_values,
            theta=list(label_list.values()),
            fill='toself',
            name=f"{f}_Average"
    )
    # Create a layout for the radar chart
    layout = go.Layout(
            polar=dict(
                radialaxis=dict(showticklabels=False, ticks='', showline=False),
                angularaxis=dict(showticklabels=True, ticks='outside', showline=True)
            ),
            showlegend=True
    )
    fig = go.Figure(data=[radar_chart_trace], layout=layout)
    # Save the figure as an image (PNG) in the output directory
    output_filename = os.path.join(output_dir, f"{f}_RadarChart.png")
    fig.write_image(output_filename)
    # Reset variables for the next song
    segment_count = 0
    label_counts = {label: 0 for label in label_list.values()}
    song_radar_values = np.zeros(len(label_list))

print(Song_list)
