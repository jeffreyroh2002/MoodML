import os
import csv
import shutil

# Replace these with the actual file paths and directory paths
csv_file = 'valence_extracted.csv'
mp3_directory = '../deam-dataset/DEAM_audio/MEMD_audio_wav'
output_directory = 'valence_dataset'

# Create the output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Read the CSV file and extract song_id and arousal/valence_mean
with open(csv_file, 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        song_id = row['song_id']
        valence_mean = row['valence_mean']

        # Create a directory for the specific arousal/valence_mean if it doesn't exist
        valence_directory = os.path.join(output_directory, f'valence_{valence_mean}')
        if not os.path.exists(valence_directory):
            os.makedirs(valence_directory)

        # Get the source mp3 file path
        mp3_file_path = os.path.join(mp3_directory, f'{song_id}.wav')

        # Check if the mp3 file exists
        if os.path.exists(mp3_file_path):
            # Copy the mp3 file to the corresponding arousal/valence directory
            destination_file_path = os.path.join(valence_directory, f'{song_id}.wav')
            shutil.copyfile(mp3_file_path, destination_file_path)

print("MP3 files copied to directories based on valence_mean.")