import os
import subprocess

# Specify the directory containing your WAV audio files
audio_directory = '../timbre-testing-data/wav_files'

# List all the audio files in directory
audio_files = []
for root, dirs, files in os.walk(audio_directory):
    for f in os.listdir(audio_directory):
        if f.endswith('.wav'):
            audio_files.append(os.path.join(root,f))

# Loop through each audio file and perform source separation for vocals
for audio_file in audio_files:
    # Construct the full path to the input audio file
    input_audio_path = audio_file

    # Perform source separation with Demucs to extract vocals
    subprocess.run(['demucs', '--two-stems=vocals', input_audio_path])

print('Extraction of vocals completed.')
