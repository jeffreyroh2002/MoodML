import os
import sys
from pytube import YouTube
from moviepy.editor import *

def download_audio_from_youtube(link, output_dir):
    try:
        yt = YouTube(link, use_oauth=True, allow_oauth_cache=True)
        stream = yt.streams.filter(only_audio=True, file_extension='mp4').first()
        if not stream:
            print(f"No audio stream found for: {link}")
            return

        audio_file = stream.download(output_path=output_dir)
        return audio_file
    except Exception as e:
        print(f"Error downloading audio from {link}: {e}")
        return None

def convert_to_wav(input_file, output_file):
    try:
        audio_clip = AudioFileClip(input_file)
        audio_clip.write_audiofile(output_file, codec="pcm_s16le")
        audio_clip.close()
    except Exception as e:
        print(f"Error converting to WAV: {e}")

def main(input_file_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(input_file_path, 'r') as file:
        links = file.read().splitlines()

    for link in links:
        audio_file = download_audio_from_youtube(link, output_dir)
        if audio_file:
            output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(audio_file))[0]}.wav")
            convert_to_wav(audio_file, output_file)
            os.remove(audio_file)
            print(f"Downloaded and converted {link} to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python download_youtube_audio.py input_file output_dir")
        sys.exit(1)

    input_file_path = sys.argv[1]
    output_dir = sys.argv[2]
    main(input_file_path, output_dir)
