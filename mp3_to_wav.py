from pydub import AudioSegment
import os

# Specify input directory containing MP3 files and output directory for WAV files
input_dir = 'DEAM_dataset/DEAM_audio'
output_dir = 'DEAM_dataset/DEAM_audio_wav'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Convert MP3 files to WAV
for mp3_file in os.listdir(input_dir):
    if mp3_file.endswith('.mp3'):
        mp3_path = os.path.join(input_dir, mp3_file)
        wav_filename = os.path.splitext(mp3_file)[0] + '.wav'
        wav_path = os.path.join(output_dir, wav_filename)

        # Load MP3 and save as WAV
        sound = AudioSegment.from_mp3(mp3_path)
        sound.export(wav_path, format='wav')

print("Conversion complete.")