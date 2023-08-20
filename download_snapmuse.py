from selenium import webdriver
import os

# Define the moods and the number of audio files to download for each mood
moods = ['Upbeat', 'Happy', 'Romantic', 'Sad', 'Calm', 'Chill', 'Dreamy', 'Relax', 'Energetic', 'Angry', 'Tense', 'Groovy']  # Add more moods as needed
num_files_per_mood = 10

# Specify the directory where you want to save the downloaded files
download_directory = 'audio_files'
os.makedirs(download_directory, exist_ok=True)

# Initialize the web driver (replace 'chromedriver.exe' with the path to your web driver executable)
driver = webdriver.Chrome()

# Loop through each mood
for mood in moods:
    print(f"Downloading {num_files_per_mood} files for '{mood}' mood...")

    # Create a subdirectory for the current mood
    mood_directory = os.path.join(download_directory, mood)
    os.makedirs(mood_directory, exist_ok=True)

    # Visit the mood's URL
    driver.get(f"https://snapmuse.com/moods/{mood}")
    
    # Locate and click the audio file download links
    for i in range(1, num_files_per_mood + 1):
        try:
            # Modify the file extension from .mp3 to .wav in the XPath
            download_button = driver.find_element_by_xpath(f"//a[contains(text(), '{mood}-{i}.wav')]")
            download_button.click()
            print(f"Downloaded {mood}-{i}.wav")
        except:
            print(f"Failed to download {mood}-{i}.wav")

print("Download complete!")

# Close the web driver
driver.quit()