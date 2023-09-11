import os
import subprocess

# Specify the directory containing your WAV audio files
audio_directory = 'dataset_lyrics/Angry'

# List all audio files in the input directory
audio_files = [f for f in os.listdir(audio_directory) if f.endswith('.wav')]

# Loop through each audio file and perform source separation for vocals
for audio_file in audio_files:
    # Construct the full path to the input audio file
    input_audio_path = os.path.join(audio_directory, audio_file)

    # Perform source separation with Demucs to extract vocals
    subprocess.run(['demucs', '--two-stems=vocals', input_audio_path])

print('Extraction of vocals completed.')