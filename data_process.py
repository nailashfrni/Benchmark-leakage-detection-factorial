import json
import itertools
import argparse

'''
{
   'option': {
        'A': 'Formed by mesenchymal proliferation',
        'B': 'Appears in the 4th week of human embryo',
        'C': 'Branial grooves are between adjacent branchial arches',
        'D': 'There are 5 pairs of branchial arches in total'
    },
    'question': 'Which of the following descriptions of branchial arches is incorrect'
}
'''
parser = argparse.ArgumentParser(prog='data_process', description='')
parser.add_argument("--data_dir", type=str)
parser.add_argument("--save_dir", type=str)
args = parser.parse_args()
with open(args.data_dir, 'r', encoding="utf8") as file:
    data_list = json.load(file)

# Define options
chars = ['A', 'B', 'C', 'D']

# Use itertools.permutations to generate all permutations
permutations_list = list(itertools.permutations(chars))
result = []

for index, row in enumerate(data_list):

    for perm in permutations_list:
        instruction = {
            "id": row['id'],
            "group": row['group'],
            "subject": row['subject'],
            "instruction":
f"""
{row['question']}:
A:{row['option'][perm[0]]}
B:{row["option"][perm[1]]}
C:{row["option"][perm[2]]} 
D:{row["option"][perm[3]]}
""",
        }
        result.append(instruction)

with open(f"{args.save_dir}/permutations_data.json", 'w', encoding='utf8') as json_file:
    json.dump(result, json_file, indent=4, ensure_ascii=False)
