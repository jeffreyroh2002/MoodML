import os
import shutil

# Define the source directory where the files are currently located
source_directory = "../vocal_timbre_analysis/yt_dataset/urls/wav_files"

# Define the destination directory where you want to move the files
destination_directory = "../vocal_timbre_analysis/yt_dataset/voice_extracted"

# Walk through the source directory and its subdirectories
for root, _, files in os.walk(source_directory):
    for file in files:
        if file == "vocals.wav":
            # Get the parent directory name
            parent_dir = os.path.basename(os.path.dirname(root))

            # Create the new filename with the parent directory name
            new_filename = f"{parent_dir}-{file}"

            # Create the full paths for source and destination
            source_path = os.path.join(root, file)
            destination_path = os.path.join(destination_directory, new_filename)

            # Rename and move the file
            shutil.move(source_path, destination_path)

print("Files moved and renamed successfully.")