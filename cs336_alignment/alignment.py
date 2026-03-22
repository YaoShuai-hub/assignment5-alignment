import torch
import torch.nn.functional as F

def tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer):
    all_input_ids = []
    all_response_masks = []
    max_len = 0
    for p, o in zip(prompt_strs, output_strs):
        p_ids = tokenizer(p, add_special_tokens=False).input_ids
        o_ids = tokenizer(o, add_special_tokens=False).input_ids
        full_ids = p_ids + o_ids
        max_len = max(max_len, len(full_ids))
        mask = [0]*len(p_ids) + [1]*len(o_ids)
        all_input_ids.append(full_ids)
        all_response_masks.append(mask)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    batch_input_ids = []
    batch_labels = []
    batch_response_mask = []
    for ids, mask in zip(all_input_ids, all_response_masks):
        pad_len = max_len - len(ids)
        if tokenizer.padding_side == "left":
            padded_ids = [pad_id]*pad_len + ids
            padded_mask = [0]*pad_len + mask
        else:
            padded_ids = ids + [pad_id]*pad_len
            padded_mask = mask + [0]*pad_len
        batch_input_ids.append(padded_ids[:-1])
        batch_labels.append(padded_ids[1:])
        batch_response_mask.append(padded_mask[1:])
    return {
        "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
        "labels": torch.tensor(batch_labels, dtype=torch.long),
        "response_mask": torch.tensor(batch_response_mask, dtype=torch.long)
    }

def compute_group_normalized_rewards(reward_fn, rollout_responses, repeated_ground_truths, group_size, advantage_eps, normalize_by_std):
    raw_rewards = []
    for r, gt in zip(rollout_responses, repeated_ground_truths):
        scores = reward_fn(r, gt)
        raw_rewards.append(scores["reward"])
    raw_rewards_tensor = torch.tensor(raw_rewards, dtype=torch.float32)
    reshaped_rewards = raw_rewards_tensor.view(-1, group_size)
    mean_rewards = reshaped_rewards.mean(dim=1, keepdim=True)
    std_rewards = reshaped_rewards.std(dim=1, unbiased=True, keepdim=True)
    if normalize_by_std:
        normalized_rewards = (reshaped_rewards - mean_rewards) / (std_rewards + advantage_eps)
    else:
        normalized_rewards = reshaped_rewards - mean_rewards
    return normalized_rewards.view(-1), raw_rewards_tensor, {}

def compute_naive_policy_gradient_loss(raw_rewards_or_advantages, policy_log_probs):
    return - (raw_rewards_or_advantages.view(-1, 1) * policy_log_probs)

def compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange):
    ratio = torch.exp(policy_log_probs - old_log_probs)
    adv = advantages.view(-1, 1)
    pg1 = -adv * ratio
    pg2 = -adv * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
    loss = torch.max(pg1, pg2)
    return loss, {}

def compute_policy_gradient_loss(policy_log_probs, loss_type, raw_rewards=None, advantages=None, old_log_probs=None, cliprange=None):
    if loss_type == "no_baseline":
        return compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs), {}
    elif loss_type == "reinforce_with_baseline":
        return compute_naive_policy_gradient_loss(advantages, policy_log_probs), {}
    elif loss_type == "grpo_clip":
        return compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)
    raise ValueError(loss_type)

def grpo_microbatch_train_step(policy_log_probs, response_mask, gradient_accumulation_steps, loss_type, raw_rewards=None, advantages=None, old_log_probs=None, cliprange=None):
    token_loss, meta = compute_policy_gradient_loss(policy_log_probs, loss_type, raw_rewards, advantages, old_log_probs, cliprange)
    loss = masked_mean(token_loss, response_mask)
    scaled_loss = loss / gradient_accumulation_steps
    scaled_loss.backward()
    return loss, meta

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    return -torch.sum(probs * log_probs, dim=-1)

def get_response_log_probs(model, input_ids, labels, return_token_entropy):
    # eval doesn't change anything for us unless there is dropout, but keep it
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids)
    logits = outputs.logits
    logits = logits.float() # Convert to float before log_softmax
    log_probs_all = torch.nn.functional.log_softmax(logits, dim=-1)
    log_probs = torch.gather(log_probs_all, dim=2, index=labels.unsqueeze(-1)).squeeze(-1)
    
    result = {"log_probs": log_probs.to(torch.float32)}
    if return_token_entropy:
        result["token_entropy"] = compute_entropy(logits).to(torch.float32)
    return result

def masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    if dim is None:
        return (tensor * mask).sum() / mask.sum()
    return (tensor * mask).sum(dim=dim) / mask.sum(dim=dim)

def masked_normalize(tensor: torch.Tensor, mask: torch.Tensor, dim: int | None = None, normalize_constant: float = 1.0) -> torch.Tensor:
    if dim is None:
        return (tensor * mask).sum() / normalize_constant
    return (tensor * mask).sum(dim=dim) / normalize_constant

def sft_microbatch_train_step(policy_log_probs, response_mask, gradient_accumulation_steps, normalize_constant=1.0):
    batch_size = policy_log_probs.size(0)
    loss = masked_normalize(-policy_log_probs, response_mask, dim=None, normalize_constant=normalize_constant)
    loss = loss / batch_size
    loss = loss / gradient_accumulation_steps
    loss.backward()
    return loss, {"loss": loss.detach()}
