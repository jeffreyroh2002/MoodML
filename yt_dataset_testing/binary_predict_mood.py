import json
import numpy as np
from tensorflow import keras
import plotly.graph_objs as go
import os

# Load the saved model
arousal_model_path = "../mood_classification/results/909_PCRNN_2D_binary_arousal_dataset_with_lyrics_3sec/saved_model"
valence_model_path = "../mood_classification/results/909_PCRNN_2D_binary_valence_dataset_with_lyrics_3sec/saved_model"
test_data_path = "p4_8songs_3sec.json"
arousal_mfcc_path = "../binary_classification/arousal_dataset_lyrics.json"
valence_mfcc_path = "../binary_classification/valence_dataset_lyrics.json"
output_dir = "new_radar_results/new_radar_910_lyrics_binaryCombined"  # Directory to save individual radar chart images

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

def combine_predictions(a_pred_class_indices, v_pred_class_indices):
    mood_labels = []

    for a_index, v_index in zip(a_pred_class_indices, v_pred_class_indices):
        if a_index == 1 and v_index == 1:
            mood_labels.append("Bright")
        elif a_index == 1 and v_index == 0:
            mood_labels.append("Angry")
        elif a_index == 0 and v_index == 1:
            mood_labels.append("Relaxed")
        elif a_index == 0 and v_index == 0:
            mood_labels.append("Melancholic")

    return mood_labels

def create_radar_chart(song_name, mood_labels):
    # Define the categories for the radar chart (Bright, Melancholic, Relaxed, Angry)
    categories = ["Bright", "Melancholic", "Relaxed", "Angry"]

    # Calculate the average mood label for the song
    mood_counts = [mood_labels.count("Bright"), mood_labels.count("Melancholic"), mood_labels.count("Relaxed"), mood_labels.count("Angry")]
    print(song_name)
    print(mood_counts)
    total_segments = len(mood_labels)
    print(total_segments)
    average_mood_probabilities = [count / total_segments for count in mood_counts]

    # Create the radar chart trace
    radar_chart = go.Figure(data=go.Scatterpolar(
        r=average_mood_probabilities,
        theta=categories,
        fill='toself'
    ))

    # Set the title and save the radar chart as an image
    radar_chart.update_layout(
        title=f'Mood Radar Chart for {song_name}',
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]  # Adjust the range as needed
            )
        )
    )

    radar_chart.write_image(os.path.join(output_dir, f'{song_name}_radar_chart.png'))

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

# Use the combine_predictions function to get mood labels for each segment
combined_mood_labels = combine_predictions(a_pred_class_indices, v_pred_class_indices)

# Define your label list mapping class indices to labels
label_list = {}
for i, label in enumerate(combined_mood_labels):
    label_list[i] = label

unique_songs = set(filenames)  # Assuming filenames contain the song names
for song_name in unique_songs:
    segments_for_song = [combined_mood_labels[i] for i, filename in enumerate(filenames) if filename == song_name]
    
    # Create and save the radar chart for the song
    create_radar_chart(song_name, segments_for_song)
