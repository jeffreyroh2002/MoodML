import json
import numpy as np
from tensorflow import keras
import plotly.graph_objs as go
import os

# Load the saved model
arousal_model_path = "../mood_classification/results/909_PCRNN_2D_binary_arousal_dataset_with_lyrics_3sec/saved_model"
valence_model_path = "../mood_classification/results/909_PCRNN_2D_binary_valence_dataset_with_lyrics_3sec/saved_model"
test_data_path = "json_files/p4_8songs_3sec.json"
arousal_mfcc_path = "../binary_classification/json_files/arousal_dataset_lyrics.json"
valence_mfcc_path = "../binary_classification/json_files/valence_dataset_lyrics.json"
output_dir = "new_radar_results/new_radar_909_lyrics_binaryCombined"  # Directory to save individual radar chart images

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

a_mfcc_labels = load_mfcc_labels(arousal_mfcc_path)
v_mfcc_labels = load_mfcc_labels(valence_mfcc_path)

arousal_model = keras.models.load_model(arousal_model_path)
valence_model = keras.models.load_model(valence_model_path)

# Make predictions
a_pred = arousal_model.predict(X_test)
v_pred = valence_model.predict(X_test)

# If you have a classification task, you can get the predicted class indices:
a_pred_class_indices = np.argmax(a_pred, axis=1)
v_pred_class_indices = np.argmax(v_pred, axis=1)

# Define your label list mapping class indices to labels
a_label_list = {}
for i in range (len(a_mfcc_labels)):
    a_label_list[i] = a_mfcc_labels[i]
v_label_list = {}
for i in range (len(v_mfcc_labels)):
    a_label_list[i] = a_mfcc_labels[i]

Song_list = set(filenames)
Song_list = list(Song_list)
Sorted_Song_list = sorted(Song_list)
Song_list = {label: [] for label in Sorted_Song_list}

# Initialize variables for percentage calculation
segment_count = 0
label_counts = {label: 0 for label in a_label_list.values()}  # Use arousal labels
song_radar_values = np.zeros(len(a_label_list))  # Use arousal labels
prev_song_title = Sorted_Song_list[0][:-7]

# Define a mapping from arousal and valence labels to your combined labels
combined_labels_mapping = {
    ("high", "high"): "Bright",
    ("high", "low"): "Angry",
    ("low", "high"): "Relaxed",
    ("low", "low"): "Melancholic"
}

for f in Sorted_Song_list:
    arousal_indices = a_pred_class_indices[Song_list[f]]  # Arousal predictions
    valence_indices = v_pred_class_indices[Song_list[f]]  # Valence predictions

    # Calculate the mode of arousal and valence predictions for this song segment
    mode_arousal = np.argmax(np.bincount(arousal_indices))
    mode_valence = np.argmax(np.bincount(valence_indices))

    # Combine the arousal and valence labels to get the final label
    combined_label = combined_labels_mapping[(a_label_list[mode_arousal], v_label_list[mode_valence])]

    f_name = f[:-7]
    if f_name != prev_song_title:
        # Calculate the average radar values for the song
        avg_radar_values = song_radar_values / segment_count
        # Output the averaged radar values to the terminal
        print(f"{f_name} Average Radar Values:")
        for label, avg_value in zip(a_label_list.values(), avg_radar_values):
            print(f"{label}: {avg_value:.2f}")

        radar_chart_trace = go.Scatterpolar(
            r=avg_radar_values,
            theta=list(a_label_list.values()),  # Use arousal labels for the radar chart
            fill='toself',
            name=f"{f_name}_Average"
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
        output_filename = os.path.join(output_dir, f"{f_name}_RadarChart.png")
        fig.write_image(output_filename)

        # Reset variables for the next song
        segment_count = 0
        label_counts = {label: 0 for label in a_label_list.values()}  # Use arousal labels
        song_radar_values = np.zeros(len(a_label_list))  # Use arousal labels

        prev_song_title = f_name

    # Update label counts for percentage calculation
    label_counts[combined_label] += 1
    segment_count += 1
    # Update radar values for the current song
    song_radar_values[a_pred_class_indices[Song_list[f]]] += 1  # Use arousal labels


