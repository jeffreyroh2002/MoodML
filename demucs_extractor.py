import os
import subprocess

# Specify the parent directory containing subdirectories with WAV files
parent_directory = 'yt_dataset_testing/yt_dataset/splited_file/'

# Loop through subdirectories
for subdir in os.listdir(parent_directory):
    subdir_path = os.path.join(parent_directory, subdir)

    if os.path.isdir(subdir_path):
        # List all the audio files in the subdirectory
        audio_files = [os.path.join(subdir_path, f) for f in os.listdir(subdir_path) if f.endswith('.wav')]

        # Loop through each audio file and perform source separation for vocals
        for audio_file in audio_files:
            # Perform source separation with Demucs to extract vocals
            subprocess.run(['demucs', '--out', 'vocals', audio_file])

print('Extraction of vocals completed.')
