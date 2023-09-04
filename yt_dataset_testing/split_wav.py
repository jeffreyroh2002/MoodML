from pydub import AudioSegment
import os
import random
import librosa
import numpy as np


PATH = "yt_dataset/unknown/"
OUT_PATH = "yt_dataset/splited_file/"


if not os.path.exists(OUT_PATH):
    os.makedirs(OUT_PATH)

def split_wav_file(filename):
    f_name = filename.split('.')
    out_song_path = os.path.join(OUT_PATH,f_name[-2])
    if not os.path.exists(out_song_path):
        os.makedirs(out_song_path)
    
    audio_path = os.path.join(PATH, filename)
    data, sr = librosa.load(audio_path)
    
    length = np.arange(len(data))/float(sr)
    length = length[-1]
    split_unit = int(length // 60)

    for i in range(split_unit): 
        t1 = i * 60
        t2 = (i+1) * 60
        t1 = t1 * 1000 #Works in milliseconds
        t2 = t2 * 1000

        
        f_name = filename.split('.')
        name = f_name[-2] + "-" + str(i+1).zfill(2)
        f_name[-2] = name
        file_name = ".".join(f_name)
        newAudio = AudioSegment.from_wav(audio_path)
        newAudio = newAudio[t1:t2]
        newAudio.export(os.path.join(out_song_path, file_name), format="wav") #Exports to a wav file in the current path.
    
    print(filename  + " has been splitted!");
    
if __name__ == "__main__":
    for (root, dirs, files) in os.walk(PATH):
        for filename in files:
            split_wav_file(filename)
