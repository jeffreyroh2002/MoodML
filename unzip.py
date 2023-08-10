import zipfile

# Specify the path to the zip file and the directory where you want to extract the contents
zip_file_path = 'DEAM_dataset.zip'
extracted_dir = 'DEAM_dataset'

# Open the zip file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    # Extract all contents to the specified directory
    zip_ref.extractall(extracted_dir)

print("Zip file extracted successfully.")