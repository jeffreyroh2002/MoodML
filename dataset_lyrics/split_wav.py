import sys
import os
from pydub import AudioSegment

def split_wav_into_segments(input_directory, output_directory, segment_length_ms=60000):
    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # List all the WAV files in the input directory
    input_files = [f for f in os.listdir(input_directory) if f.endswith(".wav")]

    for input_file in input_files:
        input_path = os.path.join(input_directory, input_file)

        # Load the input WAV file
        audio = AudioSegment.from_wav(input_path)

        # Calculate the number of segments
        num_segments = len(audio) // segment_length_ms

        # Split the audio into 1-minute segments
        for i in range(num_segments):
            start_time = i * segment_length_ms + 3000
            end_time = (i + 1) * segment_length_ms + 3000
            segment = audio[start_time:end_time]

            # Extract the original file name without extension
            file_name_without_extension = os.path.splitext(input_file)[0]

            # Define the output file name
            output_file = os.path.join(output_directory, f'{file_name_without_extension}_{i+1:02}.wav')

            # Export the segment as a new WAV file
            segment.export(output_file, format="wav")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py input_directory output_directory")
        sys.exit(1)

    input_directory = sys.argv[1]
    output_directory = sys.argv[2]
    
    split_wav_into_segments(input_directory, output_directory)

