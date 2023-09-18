from pydub import AudioSegment
from pydub.silence import split_on_silence
import sys
import os

def remove_empty_vocal(input_audio_path, output_audio_path, min_silence_duration=1000):
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

if __name__ == "__main__":
    input_audio_path = "dataset_lyrics_extracted/Vocals/Relaxed/백예린 (Yerin Baek) - 0310  가사_01_vocals.wav"
    output_audio_path = "cleaned_audio.wav"

    # Specify the minimum duration of silence to consider as empty vocal (in milliseconds)
    min_silence_duration = 500  # Adjust this value as needed

    remove_empty_vocal(input_audio_path, output_audio_path, min_silence_duration)
