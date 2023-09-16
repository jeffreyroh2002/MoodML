import sys
import os
import numpy as np
import scipy.io.wavfile as wav

# Specify the directory containing your WAV files
vocals_directory = "../yt_dataset_testing/yt_dataset_extracted/splited_files_extracted/vocals"

# Add the directory containing the Vokaturi library to the system path
sys.path.append("api")

# Import the Vokaturi module
import Vokaturi

vokaturi_library_path = "OpenVokaturi-4-0/lib/linux/OpenVokaturi-4-0-linux.so"
Vokaturi.load(vokaturi_library_path)

def analyze_emotion(audio_data, sample_rate):
    # Initialize a Vokaturi Voice object
    voice = Vokaturi.Voice(sample_rate, len(audio_data), 1)

    # Fill the Voice object with audio data
    voice.fill_float32array(len(audio_data), audio_data)

    # Extract emotion probabilities
    quality = Vokaturi.Quality()
    emotion_probabilities = Vokaturi.EmotionProbabilities()
    voice.extract(quality, emotion_probabilities)

    # Check if the analysis is valid
    if quality.valid:
        return emotion_probabilities
    else:
        return None

# Iterate through the WAV files in the directory
for filename in os.listdir(vocals_directory):
    if filename.endswith(".wav"):
        # Construct the full path to the WAV file
        wav_file_path = os.path.join(vocals_directory, filename)

        # Load the WAV file using scipy.io.wavfile
        sample_rate, audio_samples = wav.read(wav_file_path)

        # Normalize audio samples to the range [-1, 1]
        audio_samples = audio_samples / 32768.0

        # Perform emotion analysis on the audio data
        emotion_probabilities = analyze_emotion(audio_samples, sample_rate)

        if emotion_probabilities is not None:
            # Print the emotion probabilities for the current WAV file
            print(f"Emotion probabilities for {filename}:")
            print(f"Neutrality: {emotion_probabilities.neutrality * 100}%")
            print(f"Happiness: {emotion_probabilities.happiness * 100}%")
            print(f"Sadness: {emotion_probabilities.sadness * 100}%")
            print(f"Anger: {emotion_probabilities.anger * 100}%")
            print(f"Fear: {emotion_probabilities.fear * 100}%")
            print()

# Clean up resources
Vokaturi.destroy()