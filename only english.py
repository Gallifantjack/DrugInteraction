# Define input and output file paths
input_file = 'MRCONSO.RRF'
output_file = 'MRCONSO_ENG.txt'

# Open input and output files
with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
    for line in infile:
        # Split the line by '|'
        fields = line.split('|')
        
        # Check if the second field (language) is 'ENG'
        if len(fields) > 1 and fields[1] == 'ENG':
            # Write the line to the output file
            outfile.write(line)

# Provide confirmation
print(f"Filtered 'ENG' lines written to {output_file}")
