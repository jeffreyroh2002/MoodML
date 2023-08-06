import csv
import math

input_csv_file = 'averaged_songs_300.csv'  # Replace 'input_data.csv' with the actual filename of your input CSV file
output_csv_file = 'arousal_extracted.csv'  # Replace 'output_data.csv' with the desired output filename

# Lists to store the extracted data
song_ids = []
arousal_means = []

# Read the CSV file and extract 'song_id' and 'arousal_mean' columns
with open(input_csv_file, 'r') as file:
    reader = csv.reader(file)
    header = next(reader)  # Read the header row

    # Find the indices of 'song_id' and 'arousal_mean' columns
    song_id_idx = header.index('song_id')
    arousal_mean_idx = header.index('arousal_mean')

    for row in reader:
        song_id = int(row[song_id_idx])
        arousal_mean = float(row[arousal_mean_idx])
        arousal_mean_rounded = math.floor(arousal_mean)  # Round down the value

        song_ids.append(song_id)
        arousal_means.append(arousal_mean_rounded)

# Write the extracted data to the output CSV file
with open(output_csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['song_id', 'arousal_mean'])  # Write the header
    for i in range(len(song_ids)):
        writer.writerow([song_ids[i], arousal_means[i]])

print(f"Data extracted, rounded down, and saved to {output_csv_file}")