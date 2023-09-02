import json
import os
import math
import librosa

DATASET_PATH = "../SnapMuse_dataset/SnapMuse_7"
JSON_FILE_NAME = "snapmuse_6.json"
JSON_PATH = JSON_FILE_NAME

SAMPLE_RATE = 22050
DURATION = 180 # measured in seconds for GTZAN Dataset
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

def save_mfcc(dataset_path, json_path, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
	
	#data dictionary
	data = {
		"mapping": [],
		"mfcc":[],
		"labels": []
	}

	num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)

	#need number of vectors for mfcc extraction to be equal for each segment
	expected_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length) 
	
	#loop through all genres
	for i, (dirpath, dirnames, filenames) in enumerate (os.walk(dataset_path)):
		
		# ensure we're processing a genre sub-folder level
		if dirpath is not dataset_path: 

			#save semantic label
			dirpath_components = dirpath.split("/")
			semantic_label = dirpath_components[-1]
			data["mapping"].append(semantic_label)
			print("\nProcessing {}".format(semantic_label))
			
			#process files for specific genre
			for f in filenames:

				file_path = os.path.join(dirpath, f)
				signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

				#process segments extracting mfcc and storing data
				for s in range(num_segments):
					start_sample = num_samples_per_segment * s
					finish_sample = start_sample + num_samples_per_segment

					mfcc = librosa.feature.mfcc(y = signal[start_sample:finish_sample],
													   sr = sr,
													   n_fft=n_fft,
													   n_mfcc = n_mfcc,
													   hop_length=hop_length)
					mfcc = mfcc.T

					#store mfcc for segment if it has the expected length
					if len(mfcc) == expected_num_mfcc_vectors_per_segment:
						data["mfcc"].append(mfcc.tolist())  #convert numpy array to list
						data["labels"].append(i-1)
						print("{}, segment:{}".format(file_path, s))
						
	with open(json_path, "w") as fp:
		json.dump(data, fp, indent=4)

        
if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH, num_segments=60)
