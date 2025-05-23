import json

file_path = '../sae/interpret/interpret.json'
reference_file_path = '../sae/src/top_latents.txt'
output_file_path = '../sae/src/safetylatent.txt'
score_threshold = 5
k = 32

with open(file_path, 'r') as file:
    data = json.load(file)

valid_numbers = []

for number, item in data.get('results', {}).items():
    score = item.get('score')
    if score is None or score >= score_threshold:
        valid_numbers.append(number)

with open(reference_file_path, 'r') as ref_file:
    reference_numbers = [line.strip() for line in ref_file.readlines()]

sorted_numbers = [num for num in reference_numbers if num in valid_numbers]

top_k_numbers = sorted_numbers[:k]

with open(output_file_path, 'w') as output_file:
    for number in top_k_numbers:
        output_file.write(number + '\n')

    