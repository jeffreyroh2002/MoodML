import os
import numpy as np
import scipy.io.wavfile as wav
import sys

# Append the "api" directory to the sys.path
sys.path.append("OpenVokaturi-4-0/api/")

# Now you can import your custom Vokaturi module
import Vokaturi

# Specify the directory containing your WAV files
vocals_directory = "../yt_dataset_testing/yt_dataset_extracted/splited_files_extracted/vocals"

# Initialize the Vokaturi library
vokaturi_library_path = "OpenVokaturi-4-0/lib/open/linux/OpenVokaturi-4-0-linux.so"
Vokaturi.load(vokaturi_library_path)

# Iterate through the WAV files in the directory
for filename in os.listdir(vocals_directory):
    if filename.endswith(".wav"):
        # Construct the full path to the WAV file
        wav_file_path = os.path.join(vocals_directory, filename)

        # Load the WAV file using scipy.io.wavfile
        sample_rate, audio_samples = wav.read(wav_file_path)

        # Normalize audio samples to the range [-1, 1]
        audio_samples = audio_samples / 32768.0

        # Create a Vokaturi Voice object
        voice = Vokaturi.Voice(sample_rate, len(audio_samples), 1)

        # Fill the Voice object with audio data
        voice.fill_float32array(len(audio_samples), audio_samples)

        # Define Vokaturi EmotionProbabilities
        emotion_probabilities = Vokaturi.EmotionProbabilities()

        # Extract emotion probabilities
        voice.extract(emotion_probabilities)

        # Print the emotion probabilities for the current WAV file
        print(f"Emotion probabilities for {filename}:")
        print(f"Neutrality: {emotion_probabilities.neutrality * 100}%")
        print(f"Happiness: {emotion_probabilities.happiness * 100}%")
        print(f"Sadness: {emotion_probabilities.sadness * 100}%")
        print(f"Anger: {emotion_probabilities.anger * 100}%")
        print(f"Fear: {emotion_probabilities.fear * 100}%")
        print()

        # Clean up resources for this iteration
        voice.destroy()
