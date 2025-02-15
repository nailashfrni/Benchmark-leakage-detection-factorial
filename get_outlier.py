from sklearn.ensemble import IsolationForest
import numpy as np
import json
import argparse
import tqdm

def filter_data(datas, subjects, groups):
    if subjects:
        datas = [d for d in datas if d['subject'] in subjects]
    elif groups:
        datas = [d for d in datas if d['group'] in groups]
    return datas

parser = argparse.ArgumentParser(prog='get_outlier', description='')
parser.add_argument("--logprobs_dir", type=str)
parser.add_argument("--permutations_data_dir", type=str)
parser.add_argument("--save_dir", type=str)
parser.add_argument("--method", type=str)
parser.add_argument("--prefix", type=str)
parser.add_argument("--permutation_num", type=int)
parser.add_argument(
        "--groups", 
        nargs="*",
        type=str,
        help="Select specific list of groups (optional)"
    )
parser.add_argument(
        "--subjects", 
        nargs="*",
        type=str,
        help="Select specific list of subjects (optional)"
    )
args = parser.parse_args()
thresholds = [-0.2, -0.17, -0.15]

with open(args.permutations_data_dir, 'r', encoding='utf8') as file:
    list_data = json.load(file)
with open(args.logprobs_dir, 'r', encoding='utf8') as file:
    list_logprobs = json.load(file)
with open(f'{args.prefix}/data/mmlu_3000.json', 'r', encoding='utf8') as file:
    data = json.load(file)
    data = filter_data(data, args.subjects, args.groups)
    list_ids = [d['id'] for d in data]

list_data = filter_data(list_data, args.subjects, args.groups)

list_data = [list_data[i:i + args.permutation_num] for i in range(0, len(list_data), args.permutation_num)]
list_logprobs = [list_logprobs[i:i + args.permutation_num] for i in range(0, len(list_logprobs), args.permutation_num)]

subject_suffix = f"-{args.subjects}" if args.subjects else ""
groups_suffix = f"-{args.groups}" if args.groups else ""

if args.method == "shuffled":
    leakage_info = [[], [], []]
    outliers = [[], [], []]
    for index, data in enumerate(tqdm.tqdm(list_logprobs)):
        X = np.array(data).reshape(-1, 1)
        clf = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
        clf.fit(X)
        scores = clf.decision_function(X)
        max_value_index = np.argmax(data)
        max_value_score = scores[max_value_index]
        for outlier_index, threshold in enumerate(thresholds):
            leakage = 0
            if max_value_score < threshold:
                outlier = {
                    # "index": str(index),
                    "max_value_index": str(max_value_index),
                    "id": list_data[index][max_value_index]["id"],
                    "data": list_data[index][max_value_index]["instruction"],
                    "logprobs": data[max_value_index]
                }
                outliers[outlier_index].append(outlier)
                leakage = 1
            leakage_info[outlier_index].append({
                'id': list_data[index][max_value_index]["id"],
                'leakage': leakage
            })

    for i, threshold in enumerate(thresholds):
        print(f"Threshold: {threshold}. Leakage percentage: {len(outliers[i]) / len(list_data):.2f}")
        with open(f'{args.save_dir}/outliers{threshold}{subject_suffix}{groups_suffix}.json', 'w', encoding='utf8') as json_file:
            json.dump(outliers[i], json_file, indent=4, ensure_ascii=False)
        with open(f'{args.save_dir}/leakage{threshold}{subject_suffix}{groups_suffix}.json', 'w', encoding='utf8') as json_file:
            json.dump(leakage_info[i], json_file, indent=4, ensure_ascii=False)
else:
    leakage_info = []
    outliers = []
    for index, data in enumerate(list_logprobs):
        leakage = 0
        max = data[0]
        isMax = True
        for temp in data[1:]:
            if temp > max:
                isMax = False
                break
        if isMax:
            dict = {
                # "index": str(index),
                "max_value_index": str(0),
                "id": list_data[index][0]["id"],
                "data": list_data[index][0]["instruction"],
                "logprobs": data[0]
            }
            outliers.append(dict)
            leakage = 1
        leakage_info.append({
            'id': list_data[index][0]["id"],
            'leakage': leakage
        })
    with open(f'{args.save_dir}/outliers_max{subject_suffix}{groups_suffix}.json', 'w', encoding='utf8') as json_file:
        print(f"Leakage percentage: {len(outliers) / len(list_data):.2f}")
        json.dump(outliers, json_file, indent=4, ensure_ascii=False)
