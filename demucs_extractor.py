import os
import subprocess

# Specify the parent directory containing subdirectories with WAV files
parent_directory = 'vocal_timbre_analysis/yt_dataset/yt_dataset/urls/wav_files'

# Loop through subdirectories
for subdir in os.listdir(parent_directory):
    subdir_path = os.path.join(parent_directory, subdir)

    if os.path.isdir(subdir_path):
        # List all the audio files in the subdirectory
        audio_files = [os.path.join(subdir_path, f) for f in os.listdir(subdir_path) if f.endswith('.wav')]

        # Loop through each audio file and perform source separation for vocals
        for audio_file in audio_files:
            # Specify the output path for the extracted vocals
            vocal_output_path = os.path.splitext(audio_file)[0] + '_vocals.wav'
            
            # Perform source separation with Demucs to extract vocals
            subprocess.run(['demucs', '--two-stems=vocals', audio_file, '-o', vocal_output_path])

print('Extraction of vocals completed.')