import json
import torch
import random
import re
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

class PackedSFTDataset(Dataset):
    def __init__(self, data):
        self.examples = data
        
    def __len__(self):
        return len(self.examples)
        
    def __getitem__(self, idx):
        return self.examples[idx]

def get_packed_sft_dataset_impl(tokenizer, dataset_path, seq_length, shuffle):
    lines = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                lines.append(json.loads(line))
    
    if shuffle:
        random.seed(42)
        random.shuffle(lines)
                
    all_ids = []
    for item in lines:
        prompt = item["prompt"]
        response = item["response"]
        raw = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{prompt}\n\n### Response:\n{response}"
        out = tokenizer(raw).input_ids + [tokenizer.eos_token_id]
        all_ids.extend(out)
        
    num_chunks = (len(all_ids) - 1) // seq_length
    
    examples = []
    for i in range(num_chunks):
        start = i * seq_length
        end = start + seq_length + 1
        chunk = all_ids[start:end]
        examples.append({
            "input_ids": torch.tensor(chunk[:-1], dtype=torch.long),
            "labels": torch.tensor(chunk[1:], dtype=torch.long)
        })
    return PackedSFTDataset(examples)

def iterate_batches_impl(dataset, batch_size, shuffle, drop_last=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

def parse_mmlu_response_impl(mmlu_example, model_output):
    match = re.search(r'(?i)answer is\s*([A-D])\b', model_output)
    if match:
        return match.group(1).upper()
    return None

def parse_gsm8k_response_impl(model_output):
    matches = re.findall(r'[-+]?\d+(?:,\d{3})*(?:\.\d+)?', model_output)
    if not matches:
        return None
    return matches[-1]

def compute_per_instance_dpo_loss_impl(
    lm,
    lm_ref,
    tokenizer,
    beta,
    prompt,
    response_chosen,
    response_rejected,
):
    prompt_formatted = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{prompt}\n\n### Response:\n"

    p_ids = tokenizer(prompt_formatted).input_ids
    w_ids = tokenizer(prompt_formatted + response_chosen).input_ids + [tokenizer.eos_token_id]
    l_ids = tokenizer(prompt_formatted + response_rejected).input_ids + [tokenizer.eos_token_id]
    p_len = len(p_ids)

    def get_logp(m, ids):
        t = torch.tensor([ids], dtype=torch.long, device=m.device)
        with torch.no_grad(): 
            l = m(t).logits[0, :-1]
        lp = F.log_softmax(l.float(), dim=-1)
        probs = torch.gather(lp, 1, t[0, 1:].unsqueeze(-1)).squeeze(-1)
        return probs[p_len-1:].sum()

    pi_w = get_logp(lm, w_ids)
    pi_l = get_logp(lm, l_ids)
    ref_w = get_logp(lm_ref, w_ids)
    ref_l = get_logp(lm_ref, l_ids)

    diff = (pi_w - ref_w) - (pi_l - ref_l)
    return -F.logsigmoid(beta * diff)
