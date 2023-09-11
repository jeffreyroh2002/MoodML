import os
import subprocess

# Specify the directory containing your WAV audio files
audio_directory = 'dataset_lyrics/Angry'

# Specify the directory where you want to save the separated files
output_directory = 'dataset_lyrics_extracted/'

# Ensure the output directory exists, create it if not
os.makedirs(output_directory, exist_ok=True)

# List all audio files in the input directory
audio_files = [f for f in os.listdir(audio_directory) if f.endswith('.wav')]

# Loop through each audio file and perform source separation for vocals
for audio_file in audio_files:
    # Construct the full path to the input audio file
    input_audio_path = os.path.join(audio_directory, audio_file)

    # Perform source separation with Demucs to extract vocals
    subprocess.run(['demucs', '--two-stems=vocals', input_audio_path])

    # Move the separated vocals file to the output directory
    vocals_file = os.path.join('separated', 'hdemucs', audio_file.replace('.wav', ''), 'vocals.wav')

    os.rename(vocals_file, os.path.join(output_directory, audio_file.replace('.wav', '_vocals.wav')))

    # Clean up the 'separated' directory
    subprocess.run(['rm', '-r', 'separated'])

print('Extraction of vocals completed.')