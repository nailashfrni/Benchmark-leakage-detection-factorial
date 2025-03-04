python data_process.py  --data_dir data --filename clean_peft_dataset.json --save_dir data --fine_tune_type cpt
python Benchmark-leakage-detection-factorial/inference_logprobs.py --base_model_dir unsloth/Qwen2.5-0.5B-Instruct --adapter_dir nailashfrni/qwen0.5b-ift-mmlu-lora-3.0 --permutations_data_dir /kaggle/working/Benchmark-leakage-detection-factorial/data/permutations_data_cpt_clean_peft_dataset.json --save_dir data --fine_tune_type cpt --checkpoint_epoch 1
python Benchmark-leakage-detection-factorial/get_outlier.py --logprobs_dir result/logprobs/qwen-peft/logprobs_cp-epoch-1.json --permutations_data_dir /kaggle/working/Benchmark-leakage-detection-factorial/data/permutations_data_clean_peft_dataset.json --save_dir data --method shuffled --permutation_num 24 --prefix /kaggle/working/Benchmark-leakage-detection-factorial

python get_outlier.py --permutations_data_dir data/permutations_data_clean_peft_dataset.json --save_dir result/outliers/ift-newsample --method shuffled --permutation_num 24 --prefix . --logprobs_dir result/logprobs/qwen0.5-ift-newsample/logprobs_cp-epoch-1.json



python get_outlier.py --permutations_data_dir data/permutations_data_clean_peft_dataset.json --save_dir result/outliers/ift-newsample --method not_shuffled --permutation_num 24 --prefix . --logprobs_dir result/logprobs/qwen0.5-ift-newsample/logprobs_cp-epoch-1.json