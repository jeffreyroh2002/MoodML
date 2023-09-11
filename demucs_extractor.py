import torchaudio
import demucs

# Load the Demucs model (e.g., 'htdemucs' for the default model)
model = demucs.models.HybridModel('htdemucs')

# Specify the input audio file
input_audio_file = 'POWER.wav'

# Load the input audio file
waveform, sample_rate = torchaudio.load(input_audio_file)

# Perform source separation
sources = model.separate(waveform)

# Define the output directory where separated sources will be saved
output_directory = 'output_sources/'

# Save the separated sources to individual audio files
for source_name, source_waveform in sources.items():
    output_file = f'{output_directory}{source_name}.wav'
    torchaudio.save(output_file, source_waveform, sample_rate)

print('Source separation completed and files saved.')
