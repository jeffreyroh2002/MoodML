import os
import subprocess

# Specify the parent directory containing subdirectories with WAV files
parent_directory = 'yt_dataset_testing/yt_dataset/splited_file/'

# Specify the output directory for extracted vocal files
output_directory = 'yt_dataset_testing/yt_dataset/splited_file_extracted'

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
            # Specify the output path for the extracted vocal file
            output_audio_path = os.path.join(output_directory, os.path.basename(audio_file)[:-4] + '_vocals.wav')

            # Perform source separation with Demucs to extract vocals and specify the output directory
            subprocess.run(['demucs', '--out', 'vocals', '--outdir', output_directory, audio_file])

print('Extraction of vocals completed.')
