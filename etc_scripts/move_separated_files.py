import os
import shutil

FILE_PATH = "../yt_dataset_testing/yt_dataset/splited_files_extracted/htdemucs"

vocal_files = []
background_files = []

for root, dirs, files in os.walk(FILE_PATH):
    if root is not FILE_PATH:
        vocal = os.path.join(root, "vocals.wav")
        background = os.path.join(root, "no_vocals.wav")
        file_name = root.split('/')[-1]
        shutil.move(vocal, "../yt_dataset_testing/yt_dataset/splited_files_extracted/" + file_name + "_vocals.wav")
        shutil.move(background, "../yt_dataset_testing/yt_dataset/splited_files_extracted/" + file_name + "_no_vocals.wav")

print("Files have successfully moved")
