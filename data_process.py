import json
import itertools
import argparse

'''
{
    'question': 'Which of the following descriptions of branchial arches is incorrect'
    'choices': ['Formed by mesenchymal proliferation',
        'Appears in the 4th week of human embryo',
        'Branial grooves are between adjacent branchial arches',
        'There are 5 pairs of branchial arches in total']
}
'''
parser = argparse.ArgumentParser(prog='data_process', description='')
parser.add_argument("--data_dir", type=str)
parser.add_argument("--filename", type=str)
parser.add_argument("--save_dir", type=str)
parser.add_argument("--fine_tune_type", type=str) # either ift or cpt
args = parser.parse_args()
with open(f'{args.data_dir}/{args.filename}', 'r', encoding="utf8") as file:
    data_list = json.load(file)

def prompt_ift(row, perm):
    return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Answer the following multiple choice question.

### Input:
{row['question']}:
A. {row['choices'][perm[0]]}
B. {row["choices"][perm[1]]}
C. {row["choices"][perm[2]]} 
D. {row["choices"][perm[3]]}

### Response:
"""

def prompt_cpt(row, perm):
    return f"""Answer the following multiple choice question.
### Question: {row['question']}
A. {row['choices'][perm[0]]}
B. {row["choices"][perm[1]]}
C. {row["choices"][perm[2]]} 
D. {row["choices"][perm[3]]}

### Answer:"""

# Define options
chars = [0, 1, 2, 3]   # ['A', 'B', 'C', 'D']

# Use itertools.permutations to generate all permutations
permutations_list = list(itertools.permutations(chars))
result = []

for index, row in enumerate(data_list):

    for perm in permutations_list:
        instruction = {
            "id": row['id'],
            # "group": row['group'],
            # "subject": row['subject'],
            "instruction": prompt_ift(row, perm) if args.fine_tune_type == 'ift'
                            else prompt_cpt(row, perm)
        }
        result.append(instruction)

with open(f"{args.save_dir}/permutations_data_{args.fine_tune_type}_{args.filename}", 'w', encoding='utf8') as json_file:
    json.dump(result, json_file, indent=4, ensure_ascii=False)
