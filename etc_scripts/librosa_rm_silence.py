import librosa
import soundfile as sf
import numpy as np

# Load the input WAV file using librosa
input_file = "../vocal_timbre_analysis/yt_dataset/voice_original/smo_vocal/I-vocals.wav"
y, sr = librosa.load(input_file, sr=None)

# Specify the minimum silence duration in seconds
min_silence_duration = 3  # 3 seconds

# Detect silent regions based on energy threshold
silent_regions = librosa.effects.split(y, top_db=-22, frame_length=2048, hop_length=512)

# Extract non-silent segments
non_silent_audio = []
for start, end in silent_regions:
    non_silent_audio.extend(y[start:end])

# Convert the non-silent audio to a numpy array
non_silent_audio = np.array(non_silent_audio)

# Save the resulting audio to a new WAV file using soundfile
output_file = "../librosa_testing_I.wav"
sf.write(output_file, non_silent_audio, sr)

print("Silence longer than", min_silence_duration, "seconds removed and saved as", output_file)