import json
import os
import math
import librosa

DATASET_PATH = "yt_dataset/splited_files_extracted/background"
JSON_FILE_NAME = "bg_3sec.json"
JSON_PATH = JSON_FILE_NAME

SAMPLE_RATE = 22050
DURATION = 60  # measured in seconds for GTZAN Dataset
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

def save_mfcc(dataset_path, json_path, sorted_file_path, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):

    # data dictionary
    data = {
        "mapping": [],
        "mfcc": [],
        "labels": [],
        "filenames": []  # Store the file names
    }

    num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)

    # need the number of vectors for mfcc extraction to be equal for each segment
    expected_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length)
    
    for song in sorted_file_path :
        data["mapping"].append(song)
        print("\nProcessing {}".format(song))

        for file_path in sorted_file_path[song]:
            signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

            # process segments extracting mfcc and storing data
            for s in range(num_segments):
                start_sample = num_samples_per_segment * s
                finish_sample = start_sample + num_samples_per_segment

                mfcc = librosa.feature.mfcc(y=signal[start_sample:finish_sample],
                                                sr=sr,
                                                n_fft=n_fft,
                                                n_mfcc=n_mfcc,
                                                hop_length=hop_length)
                mfcc = mfcc.T

                # store mfcc for a segment if it has the expected length
                if len(mfcc) == expected_num_mfcc_vectors_per_segment:
                    data["mfcc"].append(mfcc.tolist())  # convert numpy array to list
                    data["labels"].append(0)
                    data["filenames"].append(song)  # store the file name
                    print("{}, segment:{}".format(file_path.split("/")[-1], s))

    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)

def search_path(dataset_path):
    sorted_file_paths = {}

    for root, dirs, files in os.walk(dataset_path):
        if root is not dataset_path:
            Song_name = root.split("/")[-1]
            sorted_file_paths[Song_name] = []
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                sorted_file_paths[Song_name].append(file_path)
    
    return sorted_file_paths

def sort_path(path):
    for name in path:
        path[name] = sorted(path[name])
    
    print(path)
    return path

if __name__ == "__main__":
    file_path = search_path(DATASET_PATH)
    sorted_file_path = sort_path(file_path)
    save_mfcc(DATASET_PATH, JSON_PATH, sorted_file_path, num_segments=20)
