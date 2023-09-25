import os
import shutil

# Define the source and destination directories
source_dir = "../vocal_timbre_analysis/yt_dataset/urls/wav_files/ethereal_dreamy_wav"
destination_dir = "../vocal_timbre_analysis/yt_dataset/urls/wav_files/eth_vocal"

# Walk through the source directory
for root, dirs, files in os.walk(source_dir):
    for file in files:
        # Check if the file is named "vocals.wav"
        if file == "vocals.wav":
            # Get the parent directory name
            parent_dir_name = os.path.basename(root)
            
            # Create the new file name
            new_file_name = f"{parent_dir_name}-vocals.wav"
            
            # Create the full source and destination paths
            source_file_path = os.path.join(root, file)
            destination_file_path = os.path.join(destination_dir, new_file_name)
            
            # Copy and rename the file
            shutil.copy(source_file_path, destination_file_path)

print("Copying and renaming completed.")