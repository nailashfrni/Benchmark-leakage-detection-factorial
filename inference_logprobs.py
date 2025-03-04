from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
import json
import tqdm
import argparse

parser = argparse.ArgumentParser(prog='logprobs', description='')
parser.add_argument("--model_dir", type=str)
parser.add_argument("--permutations_data_dir", type=str)
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
parser.add_argument("--save_dir", type=str)
parser.add_argument("--fine_tune_type", type=str, default=None)
parser.add_argument("--checkpoint_epoch", type=int, default=0)
args = parser.parse_args()


if not args.fine_tune_type:
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, device_map="auto", trust_remote_code=True,
                                             torch_dtype="auto").eval()
else:
    from unsloth import FastLanguageModel
    from peft import PeftModel

    if args.fine_tune_type == 'ift':
        base_model, tokenizer = FastLanguageModel.from_pretrained(
            # "unsloth/Qwen2.5-0.5B-Instruct",
            args.model_dir,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=False,
        )
        FastLanguageModel.for_inference(base_model)
        peft_model = PeftModel.from_pretrained(
            base_model,
            "nailashfrni/qwen0.5b-ift-mmlu-lora-3.0",
            revision=f"checkpoint-epoch-{args.checkpoint_epoch}"
        )
        model = peft_model.merge_and_unload()
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model_dir, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(base_model_dir, trust_remote_code=True,
                                                    torch_dtype=torch.float16, device_map="auto")
        peft_model = PeftModel.from_pretrained(
                        base_model,
                        adapter_dir,
                        revision=f"checkpoint-epoch-{checkpoint_epoch}"
                    )
        model = peft_model.merge_and_unload()


def find_indices(lst, value):
    indices = []
    for i, elem in enumerate(lst):
        if (elem == value and len(lst[i + 1]) != 0 and lst[i + 1][0] == ".") or elem == 'A.':
            indices.append(i)
            return indices
    return indices


def score(prompt):
    with torch.no_grad():
        input_ids = tokenizer(prompt, return_tensors="pt").to(model.device).input_ids
        input_tokens = [tokenizer.decode([id]) for id in input_ids[0]]
        index = find_indices(input_tokens, 'A')
        logits = model(input_ids).logits
        all_tokens_logprobs = F.log_softmax(logits.double(), dim=2)
        input_logprobs = [all_tokens_logprobs[:, k - 1, input_ids[0, k]] for k in range(1, input_ids.shape[1])]
        input_logprobs = [input_logprobs[k].detach().cpu().numpy()[0] for k in range(len(input_logprobs))]
        del logits
        return input_tokens, input_logprobs, index[0]


def display(prompt):
    input_tokens, input_logprobs, index = score(prompt)
    all_logprobs = 0
    for i in range(index, len(input_logprobs)):
        all_logprobs = all_logprobs + input_logprobs[i]
    return all_logprobs


with open(args.permutations_data_dir, 'r', encoding='utf8') as file:
    datas = json.load(file)

# if args.subjects:
#     datas = [d for d in datas if d['subject'] in args.subjects]
# elif args.groups:
#     datas = [d for d in datas if d['group'] in args.groups]
subject_suffix = f"-{args.subjects}" if args.subjects else ""
groups_suffix = f"-{args.groups}" if args.groups else ""
cp_epoch_suffix = f"_cp-epoch-{args.checkpoint_epoch}" if (args.is_peft and args.checkpoint_epoch > 0) else ""

logprobs_list = []

for index,data in enumerate(tqdm.tqdm(datas)):

    result = display(data["instruction"])
    logprobs_list.append(result)
    torch.cuda.empty_cache()

with open(f"{args.save_dir}/logprobs{cp_epoch_suffix}{subject_suffix}{groups_suffix}.json", 'w', encoding='utf8') as json_file:
    json.dump(logprobs_list, json_file, indent=4, ensure_ascii=False)
