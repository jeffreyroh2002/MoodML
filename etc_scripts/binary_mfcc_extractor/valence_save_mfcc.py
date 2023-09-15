import json
import os
import math
import librosa

DATASET_PATH = "../../dataset_lyrics_extracted/Background"
JSON_FILE_NAME = "../../dataset_lyrics_extracted/background_valence.json"
JSON_PATH = JSON_FILE_NAME

SAMPLE_RATE = 22050
DURATION = 60
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

def map_to_binary_labels(semantic_label):
    if semantic_label in ["Bright", "Relaxed"]:
        return "high_valence"
    elif semantic_label in ["Angry", "Melancholic"]:
        return "low_valence"
    else:
        raise ValueError(f"Unknown label: {semantic_label}")

def save_mfcc(dataset_path, json_path, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):

    data = {
        "mapping": ["high_valence", "low_valence"],
        "mfcc": [],
        "labels": [],
    }

    num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    expected_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length)

    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        if dirpath is not dataset_path:
            semantic_label = dirpath.split("/")[-1]
            binary_label = map_to_binary_labels(semantic_label)
            print("\nProcessing {} ({} label)".format(semantic_label, binary_label))

            for f in filenames:
                file_path = os.path.join(dirpath, f)
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

                for s in range(num_segments):
                    start_sample = num_samples_per_segment * s
                    finish_sample = start_sample + num_samples_per_segment

                    mfcc = librosa.feature.mfcc(y=signal[start_sample:finish_sample],
                                                sr=sr,
                                                n_fft=n_fft,
                                                n_mfcc=n_mfcc,
                                                hop_length=hop_length)
                    mfcc = mfcc.T

                    if len(mfcc) == expected_num_mfcc_vectors_per_segment:
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(0 if binary_label == "high_valence" else 1)
                        print("{}, segment:{}".format(file_path, s))

    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)

if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH, num_segments=20)
