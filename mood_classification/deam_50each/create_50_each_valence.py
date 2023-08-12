import os
import csv
import math
import shutil

# Replace these with the actual file paths and directory paths
csv_file = '../../DEAM_dataset/DEAM_Annotations/static_annotations_averaged_songs_1_2000.csv'
mp3_directory = '../../DEAM_dataset/DEAM_audio_wav'
output_directory = 'valence_dataset'
max_files_per_valence = 50  # Maximum number of files to copy per valence value

# Create the output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Read the CSV file and extract song_id and arousal/valence_mean
with open(csv_file, 'r') as file:
    reader = csv.DictReader(file)
    valence_counter = {}  # To keep track of copied files for each valence value
    
    for row in reader:
        song_id = row['song_id']
        valence_mean = float(row['valence_mean'])  #unable to access specific row"
        
        # Floor the valence_mean to get a whole number
       valence_mean_floor = math.floor(valence_mean)

        # Initialize the counter for this valence value
        if valence_mean_floor not in valence_counter:
            valence_counter[valence_mean_floor] = 0
        
        # Check if the counter has reached the maximum
        if valence_counter[valence_mean_floor] >= max_files_per_valence:
            continue  # Skip copying more files for this valence value
        
        # Create a directory for the specific arousal/valence_mean if it doesn't exist
        valence_directory = os.path.join(output_directory, f'valence{valence_mean_floor}')
        if not os.path.exists(valence_directory):
            os.makedirs(valence_directory)

        # Get the source mp3 file path
        mp3_file_path = os.path.join(mp3_directory, f'{song_id}.wav')

        # Check if the mp3 file exists
        if os.path.exists(mp3_file_path):
            # Copy the mp3 file to the corresponding arousal/valence directory
            destination_file_path = os.path.join(valence_directory, f'{song_id}.wav')
            shutil.copyfile(mp3_file_path, destination_file_path)
            
            # Increment the counter for this valence value
            valence_counter[valence_mean_floor] += 1

print(f"First {max_files_per_valence} MP3 files copied to directories based on whole number valence_mean.")