import os
import requests

# Define the moods and the number of audio files to download for each mood
moods = ['Upbeat', 'Happy', 'Romantic', 'Sad', 'Calm', 'Chill', 'Dreamy', 'Relax', 'Energetic', 'Angry', 'Tense', 'Groovy']  # Add more moods as needed
num_files_per_mood = 15

# Specify the directory where you want to save the downloaded files
download_directory = 'SnapMuse_dataset'
os.makedirs(download_directory, exist_ok=True)

# Base URL for the audio files
base_url = 'https://snapmuse.com/moods/'

# Loop through each mood
for mood in moods:
    print(f"Downloading {num_files_per_mood} files for '{mood}' mood...")
    
    # Create a subdirectory for the current mood
    mood_directory = os.path.join(download_directory, mood)
    os.makedirs(mood_directory, exist_ok=True)
    
    # Download the audio files
    for i in range(1, num_files_per_mood + 1):
        file_url = f"{base_url}{mood}/{mood}-{i}.mp3"
        response = requests.get(file_url)
        
        if response.status_code == 200:
            file_path = os.path.join(mood_directory, f"{mood}-{i}.mp3")
            with open(file_path, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded {file_path}")
        else:
            print(f"Failed to download {file_url}")
            
print("Download complete!")