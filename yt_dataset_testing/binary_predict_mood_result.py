import json
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import os

# Load the saved model
arousal_model_path = "../mood_classification/results/909_PCRNN_2D_binary_arousal_dataset_with_lyrics_3sec/saved_model"
valence_model_path = "../mood_classification/results/909_PCRNN_2D_binary_valence_dataset_with_lyrics_3sec/saved_model"
test_data_path = "p4_8songs_3sec.json"
arousal_mfcc_path = "../binary_classification/arousal_dataset_lyrics.json"
valence_mfcc_path = "../binary_classification/valence_dataset_lyrics.json"
output_dir = "new_radar_results/913_binaryCombined"  # Directory to save individual radar chart images

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
    arousal = a_pred_class_indices.index(max(a_pred_class_indices))
    valence = v_pred_class_indices.index(max(v_pred_class_indices))

    if arousal == 0 and valence == 0:
        mood_labels = "Bright"
    elif arousal == 0 and valence == 1:
        mood_labels = "Angry"
    elif arousal == 1 and valence == 0:
        mood_labels = "Relaxed"
    elif arousal == 1 and valence == 1:
        mood_labels = "Melancholic"
    
    return_phrase = "Overall Mood : " + mood_labels

    return return_phrase

def create_radar_chart(song_name, mood_labels):
    # Define the categories for the radar chart (Bright, Melancholic, Relaxed, Angry)
    categories = ["Bright", "Melancholic", "Relaxed", "Angry"]
    a_labels = []
    v_labels = []

    # Calculate the average mood label for the song
    for idx in mood_labels:
        a_labels.append(a_pred_class_indices[idx])
        v_labels.append(v_pred_class_indices[idx])
    
    a_mood_counts = [a_labels.count(0), a_labels.count(1)]
    v_mood_counts = [v_labels.count(0), v_labels.count(1)]
    total_segments = len(mood_labels)
    average_a_mood_probabilities = [count / total_segments for count in a_mood_counts]
    average_v_mood_probabilities = [count / total_segments for count in v_mood_counts]
    
    
    output_phrase = combine_predictions(average_a_mood_probabilities, average_v_mood_probabilities)

    # Create a scatter plot for valence and arousal
    plt.figure(figsize=(8, 6))
    plt.scatter(average_v_mood_probabilities, average_a_mood_probabilities, color='b', marker='o', label='Data Points')

    # Add labels and title
    plt.xlabel('Valence')
    plt.ylabel('Arousal')
    plt.title('Predicted Mood for {}'.format(song_name))

    # Add the overall output phrase at the bottom
    plt.text(0.2, -0.2, output_phrase, fontsize=12, ha='left')

    # Customize the axis limits if needed
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    # Show legend
    plt.legend()

    # Display the plot
    plt.grid(True)
    plt.show()

    # Save plt
    plt.savefig(os.path.join(output_dir, song_name+".png"))



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



unique_songs = set(filenames)  # Assuming filenames contain the song names

for song_name in unique_songs:
    segments_for_song = [i for i, filename in enumerate(filenames) if filename == song_name]

    # Create and save the radar chart for the song
    create_radar_chart(song_name, segments_for_song)
