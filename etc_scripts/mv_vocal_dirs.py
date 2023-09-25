import os
import shutil

# Define the source directory
source_dir = "~/MoodML/vocal_timbre_analysis/yt_dataset/urls/wav_files"

# Define the destination directory
destination_dir = "~/MoodML/vocal_timbre_analysis/yt_dataset/voice_extracted"

# Function to copy and rename files
def copy_and_rename_files(src_dir, dest_dir):
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file == "vocals.wav":
                # Get the parent directory name
                parent_dir = os.path.basename(os.path.dirname(root))
                # Create the new directory in the destination
                new_dir = os.path.join(dest_dir, parent_dir)
                os.makedirs(new_dir, exist_ok=True)
                # Create the new file name
                new_name = f"{parent_dir}-{file}"
                old_path = os.path.join(root, file)
                new_path = os.path.join(new_dir, new_name)
                # Copy the file to the new location with the new name
                shutil.copy(old_path, new_path)
                print(f"Copied: {old_path} -> {new_path}")

# Call the function to copy and rename files
copy_and_rename_files(source_dir, destination_dir)