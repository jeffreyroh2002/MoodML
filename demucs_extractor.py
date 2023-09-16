import os
import subprocess
import shutil

# Specify the parent directory containing subdirectories with WAV files
parent_directory = '~/MoodML/yt_dataset_testing/yt_dataset/splited_file/'

# Specify the output directory for extracted vocal files
output_directory = '~/MoodML/yt_dataset_testing/yt_dataset/splited_file/extracted_vocals/'

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

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

            # Move the extracted vocal file to the output directory
            vocal_file = os.path.splitext(os.path.basename(audio_file))[0] + '_vocals.wav'
            vocal_path = os.path.join(subdir_path, vocal_file)
            shutil.move(vocal_path, os.path.join(output_directory, vocal_file))

print('Extraction of vocals completed.')
