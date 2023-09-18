import sys
import scipy.io.wavfile
import os


sys.path.append("OpenVokaturi-4-0/api/")

# Now you can import your custom Vokaturi module
import Vokaturi

print("Loading library...")
Vokaturi.load("OpenVokaturi-4-0/lib/open/linux/OpenVokaturi-4-0-linux.so")
print ("Analyzed by: %s" % Vokaturi.versionAndLicense())

# Specify the directory containing your WAV files
vocals_directory = "../yt_dataset_testing/yt_dataset_extracted/splited_files_extracted/vocals"

# Iterate through the WAV files in the directory
for file_name in os.listdir(vocals_directory):
    wav_file_path = os.path.join(vocals_directory, file_name)

    print("Reading soung file : ", file_name)
    (sample_rate, samples) = scipy.io.wavfile.read(wav_file_path)
    print("   sample rate %.3f Hz" % sample_rate)
    
    print("Allocating Vokaturi sample array...")
    buffer_length = len(samples)
    print("   %d samples, %d channels" % (buffer_length, samples.ndim))
    c_buffer = Vokaturi.float64array(buffer_length)

    if samples.ndim == 1:
        c_buffer[:] = samples[:] / 32768.0  # mono
    else:
        c_buffer[:] = 0.5*(samples[:,0]+0.0+samples[:,1]) / 32768.0  # stereo

    print("Creating VokaturiVoice...")
    voice = Vokaturi.Voice(sample_rate, buffer_length, 0)

    print("Filling VokaturiVoice with samples...")
    voice.fill_float64array(buffer_length, c_buffer)

    print("Extracting emotions from VokaturiVoice...")
    quality = Vokaturi.Quality()
    emotionProbabilities = Vokaturi.EmotionProbabilities()
    voice.extract(quality, emotionProbabilities)

    if quality.valid:
        print("Neutral: %.3f" % emotionProbabilities.neutrality)
        print("Happy: %.3f" % emotionProbabilities.happiness)
        print("Sad: %.3f" % emotionProbabilities.sadness)
        print("Angry: %.3f" % emotionProbabilities.anger)
        print("Fear: %.3f" % emotionProbabilities.fear)
    else:
        print ("Not enough sonorancy to determine emotions")

    voice.destroy()
