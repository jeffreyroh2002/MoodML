from pydub import AudioSegment
from pydub.silence import split_on_silence

# Load the input WAV file
input_file = "../vocal_timbre_analysis/yt_dataset/voice_original/eth_vocal/Billie Eilish - my future (Lyrics)-vocals.wav"
audio = AudioSegment.from_file(input_file, format="wav")

# Split the audio into non-silent segments with a minimum silence length of 3 seconds
min_silence_len = 3000  # 3 seconds in milliseconds
non_silent_segments = split_on_silence(
    audio,
    min_silence_len=min_silence_len,
    silence_thresh=-23,    # Silence threshold in dBFS (adjust as needed)
    keep_silence=100       # Amount of silence to keep at the beginning/end of segments
)

# Combine non-silent segments into a new audio
output_audio = AudioSegment.empty()
for segment in non_silent_segments:
    output_audio += segment

# Save the resulting audio to a new WAV file
output_file = "../output.wav"
output_audio.export(output_file, format="wav")

print("Silence longer than 3 seconds removed and saved as", output_file)
