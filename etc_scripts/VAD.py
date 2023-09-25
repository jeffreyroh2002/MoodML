from pydub import AudioSegment
from pydub.silence import split_on_silence
import os

def remove_empty_vocal(input_audio_path, output_audio_path, min_silence_duration=10000):
    # Load the input vocal audio file
    audio = AudioSegment.from_file(input_audio_path, format="wav")

    # Split the audio into non-silent chunks (vocal parts)
    non_silent_audio = split_on_silence(audio, min_silence_len=min_silence_duration)

    # Concatenate the non-silent vocal parts to create the cleaned audio
    cleaned_audio = AudioSegment.empty()
    for segment in non_silent_audio:
        cleaned_audio += segment

    # Export the cleaned audio to the output file
    cleaned_audio.export(output_audio_path, format="wav")

def process_directory(input_directory, output_directory, min_silence_duration=10000):
    # Traverse the input directory and its subdirectories
    for root, _, files in os.walk(input_directory):
        for file in files:
            if file.endswith(".wav"):
                input_audio_path = os.path.join(root, file)

                # Create the same directory structure in the output directory
                output_audio_dir = os.path.join(output_directory, os.path.relpath(root, input_directory))
                os.makedirs(output_audio_dir, exist_ok=True)

                # Generate the output audio path with a different name
                output_audio_path = os.path.join(output_audio_dir, file.replace(".wav", "_cleaned.wav"))

                # Process and remove empty vocals
                remove_empty_vocal(input_audio_path, output_audio_path, min_silence_duration)

if __name__ == "__main__":
    input_directory = "../vocal_timbre_analysis/yt_dataset/voice_original"  # Replace with your input directory
    output_directory = "../vocal_timbre_analysis/yt_dataset/voice_cleaned"  # Replace with your output directory

    # Specify the minimum duration of silence to consider as empty vocal (in milliseconds)
    min_silence_duration = 10000  # Adjust this value as needed

    process_directory(input_directory, output_directory, min_silence_duration)