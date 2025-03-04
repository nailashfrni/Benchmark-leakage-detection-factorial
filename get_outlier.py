from sklearn.ensemble import IsolationForest
from sklearn.metrics import cohen_kappa_score
import numpy as np
import json
import argparse
import tqdm

# def filter_data(datas, subjects, groups):
#     if subjects:
#         datas = [d for d in datas if d['subject'] in subjects]
#     elif groups:
#         datas = [d for d in datas if d['group'] in groups]
#     return datas

def display_matching_accuracy(list1, list2):
    assert len(list1) == len(list2), "Lists must have the same length"
    matches = sum(a == b for a, b in zip(list1, list2))
    print(f'Matching Accuracy: {matches / len(list1)} ({matches}/{len(list1)})')
    return matches, matches / len(list1)

def matching_counter(list1, list2, is_leakage):
    assert len(list1) == len(list2), "Lists must have the same length"
    symbol = 1 if is_leakage else 0
    matches = sum(a == b == symbol for a, b in zip(list1, list2))
    return matches

def hamming_distance(list1, list2):
    assert len(list1) == len(list2), "Lists must have the same length"
    count = 0
    for i in range(len(list1)):
        if list1[i] != list2[i]:
            count += 1
    return count

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
with open(f'{args.prefix}/data/peft_dataset.json', 'r', encoding='utf8') as file:
    dataset = json.load(file)
    # data = filter_data(data, args.subjects, args.groups)

# list_data = filter_data(list_data, args.subjects, args.groups)

list_data = [list_data[i:i + args.permutation_num] for i in range(0, len(list_data), args.permutation_num)]
list_logprobs = [list_logprobs[i:i + args.permutation_num] for i in range(0, len(list_logprobs), args.permutation_num)]

subject_suffix = f"-{args.subjects}" if args.subjects else ""
groups_suffix = f"-{args.groups}" if args.groups else ""
y_true = [d['label'] for d in dataset]

checkpoint_epoch = 0
if 'cp' in args.logprobs_dir:
    checkpoint_epoch = int(args.logprobs_dir.split("-")[-1].replace(".json", ""))
cp_epoch_suffix = f"_cp-epoch-{checkpoint_epoch}" if (checkpoint_epoch > 0) else ""

if args.method == "shuffled":
    leakage_info = [[], [], []]
    outliers = [[], [], []]
    y_pred = [[], [], []]
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
                    "logprobs": data[max_value_index],
                    "label": dataset[index]['label']
                }
                outliers[outlier_index].append(outlier)
                leakage = 1
            leakage_info[outlier_index].append({
                'id': list_data[index][max_value_index]["id"],
                'leakage': leakage,
                "label": dataset[index]['label']
            })
            y_pred[outlier_index].append(leakage)

    for i, threshold in enumerate(thresholds):
        kappa = cohen_kappa_score(y_true, y_pred[i])
        print(f"Threshold: {threshold}. Kappa Score: {kappa}. Leakage percentage: {len(outliers[i]) / len(list_data):.2f}")
        with open(f'{args.save_dir}/outliers{threshold}{cp_epoch_suffix}{subject_suffix}{groups_suffix}.json', 'w', encoding='utf8') as json_file:
            json.dump(outliers[i], json_file, indent=4, ensure_ascii=False)
        with open(f'{args.save_dir}/leakage{threshold}{cp_epoch_suffix}{subject_suffix}{groups_suffix}.json', 'w', encoding='utf8') as json_file:
            json.dump(leakage_info[i], json_file, indent=4, ensure_ascii=False)
else:
    leakage_info = []
    outliers = []
    y_pred = []
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
                "logprobs": data[0],
                "label": dataset[index]['label']
            }
            outliers.append(dict)
            leakage = 1
        leakage_info.append({
            'id': list_data[index][0]["id"],
            'leakage': leakage,
            "label": dataset[index]['label']
        })
        y_pred.append(leakage)

    with open(f'{args.save_dir}/outliers_max{cp_epoch_suffix}{subject_suffix}{groups_suffix}.json', 'w', encoding='utf8') as json_file:
        json.dump(outliers, json_file, indent=4, ensure_ascii=False)
    display_matching_accuracy(y_true, y_pred)
    both_leakage = matching_counter(y_true, y_pred, True)
    both_not_leakage = matching_counter(y_true, y_pred, False)
    kappa = cohen_kappa_score(y_true, y_pred)
    print(f"Kappa Score: {kappa}. Leakage percentage: {len(outliers) / len(list_data):.2f}")
    print(f'Both Leakage: {both_leakage}')
    print(f'Both Not Leakage: {both_not_leakage}')
    print('Hamming distance:', hamming_distance(y_true, y_pred))
