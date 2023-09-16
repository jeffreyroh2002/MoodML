import os
import shutil

# Define the source directory
source_directory = "../yt_dataset_testing/yt_dataset/splited_files_extracted"

# Define the destination directories
vocals_directory = "../yt_dataset_testing/yt_dataset/splited_files_extracted/vocals"
background_directory = "../yt_dataset_testing/yt_dataset/splited_files_extracted/background"

# Expand the tilde (~) to the user's home directory
source_directory = os.path.expanduser(source_directory)
vocals_directory = os.path.expanduser(vocals_directory)
background_directory = os.path.expanduser(background_directory)

# Iterate through the source directory
for root, _, files in os.walk(source_directory):
    for filename in files:
        if filename == "no_vocals.wav":
            # For non-vocals, rename and move to the background directory
            new_filename = os.path.basename(root) + "_no_vocals.wav"
            destination = os.path.join(background_directory, new_filename)
        elif filename == "vocals.wav":
            # For vocals, rename and move to the vocals directory
            new_filename = os.path.basename(root) + "_vocals.wav"
            destination = os.path.join(vocals_directory, new_filename)
        else:
            continue  # Skip files that are not vocals or no_vocals

        # Ensure the destination directory exists, create it if not
        os.makedirs(os.path.dirname(destination), exist_ok=True)

        # Construct source and destination paths
        source_path = os.path.join(root, filename)

        # Move and rename the file
        shutil.move(source_path, destination)
        print(f"Moved {source_path} to {destination}")

print("All files processed.")