python get_outlier.py --permutations_data_dir data/permutations_data_peft_dataset.json --save_dir result/outliers/peft --method shuffled --permutation_num 24 --prefix . --logprobs_dir result/logprobs/qwen-peft/logprobs_cp-epoch-7.json
python get_outlier.py --permutations_data_dir data/permutations_data_peft_dataset.json --save_dir result/outliers/peft --method shuffled --permutation_num 24 --prefix . --logprobs_dir result/logprobs/qwen-peft/logprobs_cp-epoch-9.json




python get_outlier.py --permutations_data_dir data/permutations_data_peft_dataset.json --save_dir result/outliers/peft-instruct --method not_shuffled --permutation_num 24 --prefix . --logprobs_dir result/logprobs/qwen0.5b-instruct-peft/logprobs_cp-epoch-9.json