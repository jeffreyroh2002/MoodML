import re

# Function to extract URLs from a text file and write them to another file
def extract_urls(input_file, output_file):
    with open(input_file, 'r') as file:
        text = file.read()

    # Regular expression to match URLs
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    # Find all URLs in the text
    urls = re.findall(url_pattern, text)

    # Write the URLs to the output file
    with open(output_file, 'w') as output:
        for url in urls:
            output.write(url + '\n')

# Replace 'input.txt' with the path to your input text file
input_file = 'input.txt'

# Replace 'output.txt' with the path where you want to save the URLs
output_file = 'output.txt'

# Call the function to extract and write URLs
extract_urls(input_file, output_file)