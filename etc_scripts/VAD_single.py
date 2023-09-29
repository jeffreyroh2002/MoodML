from pydub import AudioSegment
from pydub.silence import split_on_silence

def remove_empty_vocal(input_audio_path, output_audio_path, min_silence_duration=3000):
    try:
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

        print(f"Processed: {input_audio_path} -> {output_audio_path}")

    except Exception as e:
        print(f"Error processing {input_audio_path}: {e}")

if __name__ == "__main__":
    input_audio_path = "../vocal_timbre_analysis/yt_dataset/voice_original/smo_vocal/I-vocals.wav"  # Replace with your input audio file path
    output_audio_path = "../vocal_timbre_analysis/yt_dataset/voice_cleaned/I-vocals-cleaned.wav"  # Specify the output file path

    # Specify the minimum duration of silence to consider as empty vocal (in milliseconds)
    min_silence_duration = 10000  # Adjust this value as needed

    remove_empty_vocal(input_audio_path, output_audio_path, min_silence_duration)
