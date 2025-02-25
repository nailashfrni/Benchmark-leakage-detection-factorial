import json
import os
import pandas as pd

def get_group_leakage_percentage(df, group):
    return df[(df['group'] == group) & (df['leakage'] == 1)].shape[0] / df.shape[0] * 100

thresholds = [0.2, 0.17, 0.15]
original_dir = "./result/outliers/"
all_filenames = os.listdir(original_dir)
df_mmlu = pd.read_csv('data/mmlu_3000_fix.csv')[['id', 'subject', 'group']]
factorial_results = [[], [], []]

for filename in all_filenames:
    if not filename.startswith('leakage'):
        continue
    threshold = float(filename.split('-')[1])
    idx = thresholds.index(threshold)
    path = os.path.join(original_dir, filename)
    with open(path, 'r', encoding='utf-8') as file:
        temp_data = json.load(file)
        factorial_results[idx].extend(temp_data)

for i in range(len(thresholds)):
    threshold = thresholds[i]
    factorial_results[i].sort(key= lambda x: x['id'])
    df_result = pd.DataFrame(factorial_results[i])
    df_merged = df_result.merge(df_mmlu, on='id', how='left')
    updated_list = df_merged.to_dict(orient="records")

    with open(f'result/leakage_{threshold}_3000.json', 'w', encoding='utf8') as json_file:
        json.dump(updated_list, json_file, indent=4, ensure_ascii=False)

    # print statistics
    leakage_percentage = df_merged[df_merged['leakage'] == 1].shape[0] / df_merged.shape[0] * 100
    print(f'Threshold: {threshold}. Leakage: {leakage_percentage}%')
    print(f'STEM: {get_group_leakage_percentage(df_merged, "stem")}%')
    print(f'Social Sciences: {get_group_leakage_percentage(df_merged, "social_sciences")}%')
    print(f'Humanities: {get_group_leakage_percentage(df_merged, "humanities")}%')
    print(f'Other: {get_group_leakage_percentage(df_merged, "other")}%')
    print()