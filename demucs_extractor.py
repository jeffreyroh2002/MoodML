import os
import subprocess

# Specify the directory containing your WAV audio files
audio_directory = 'dataset_lyrics/Preprocessed/Angry'

# Specify the directory where you want to save the separated files
output_directory = 'dataset_lyrics_extracted/'

# Ensure the output directories exist, create them if not
os.makedirs(os.path.join(output_directory, 'angry_vocals'), exist_ok=True)
os.makedirs(os.path.join(output_directory, 'angry_background'), exist_ok=True)

# List all audio files in the input directory
audio_files = [f for f in os.listdir(audio_directory) if f.endswith('.wav')]

# Loop through each audio file and perform source separation
for audio_file in audio_files:
    # Construct the full path to the input audio file
    input_audio_path = os.path.join(audio_directory, audio_file)

    # Perform source separation with Demucs to extract all four sources
    subprocess.run(['demucs', input_audio_path])

    # Move the separated vocals and background files to their respective directories
    vocals_file = os.path.join('separated', 'hdemucs', audio_file.replace('.wav', ''), 'vocals.wav')
    background_file = os.path.join('separated', 'hdemucs', audio_file.replace('.wav', ''), 'other.wav')

    output_vocals_directory = os.path.join(output_directory, 'angry_vocals')
    output_background_directory = os.path.join(output_directory, 'angry_background')

    os.rename(vocals_file, os.path.join(output_vocals_directory, audio_file.replace('.wav', '_vocals.wav')))
    os.rename(background_file, os.path.join(output_background_directory, audio_file.replace('.wav', '_background.wav')))

    # Clean up the 'separated' directory (remove drums.wav and bass.wav)
    subprocess.run(['rm', os.path.join('separated', 'hdemucs', audio_file.replace('.wav', ''), 'drums.wav')])
    subprocess.run(['rm', os.path.join('separated', 'hdemucs', audio_file.replace('.wav', ''), 'bass.wav')])

print('Extraction completed.')
