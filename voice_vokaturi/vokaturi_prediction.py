import os
import Vokaturi
import scipy.io.wavfile as wav

# Load the Vokaturi library (make sure to specify the correct library path)
vokaturi_library_path = "OpenVokaturi-4-0/lib/linux/OpenVokaturi-4-0-linux.so"
Vokaturi.load(vokaturi_library_path)

# Initialize a Vokaturi Voice object
voice = Vokaturi.Voice()

# Set the sample rate (adjust this based on your WAV file's sample rate)
sample_rate = 44100  # Change to match your WAV file's sample rate
voice.setSampleRate(sample_rate)

# Specify the directory containing your WAV files
vocals_directory = "../yt_dataset_testing/yt_dataset_extracted/splited_files_extracted/vocals"

# Iterate through the WAV files in the directory
for filename in os.listdir(vocals_directory):
    if filename.endswith(".wav"):
        # Construct the full path to the WAV file
        wav_file_path = os.path.join(vocals_directory, filename)

        # Load the WAV file
        sample_rate, audio_samples = wav.read(wav_file_path)

        # Start recording
        voice.startRecording()

        # Add the audio samples to the Voice object
        voice.recordSamples(audio_samples)

        # Stop recording
        voice.stopRecording()

        # Analyze the emotion
        emotion = voice.emotions

        # Get emotion probabilities
        emotion_values = emotion.getEmotionProbabilities()

        # Map emotion values to labels
        emotions = ["Neutral", "Happy", "Sad", "Angry", "Fearful", "Surprised"]

        # Print the emotion probabilities for the current WAV file
        print(f"Emotion probabilities for {filename}:")
        for i, emotion_label in enumerate(emotions):
            print(f"{emotion_label}: {emotion_values[i]}")

# Clean up resources
voice.destroy()