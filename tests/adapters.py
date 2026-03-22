from __future__ import annotations

import os
from typing import Any, Callable, Literal

import torch
from torch import Tensor
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from cs336_alignment.alignment import (
    tokenize_prompt_and_output,
    compute_group_normalized_rewards,
    compute_entropy,
    get_response_log_probs,
    compute_naive_policy_gradient_loss,
    compute_grpo_clip_loss,
    compute_policy_gradient_loss,
    masked_mean,
    sft_microbatch_train_step,
    grpo_microbatch_train_step,
    masked_normalize,
)

from cs336_alignment.data_utils import (
    get_packed_sft_dataset_impl,
    iterate_batches_impl,
    parse_mmlu_response_impl,
    parse_gsm8k_response_impl,
    compute_per_instance_dpo_loss_impl,
)

def run_tokenize_prompt_and_output(*args, **kwargs): return tokenize_prompt_and_output(*args, **kwargs)
def run_compute_group_normalized_rewards(*args, **kwargs): return compute_group_normalized_rewards(*args, **kwargs)
def run_compute_entropy(*args, **kwargs): return compute_entropy(*args, **kwargs)
def run_get_response_log_probs(*args, **kwargs): return get_response_log_probs(*args, **kwargs)
def run_compute_naive_policy_gradient_loss(*args, **kwargs): return compute_naive_policy_gradient_loss(*args, **kwargs)
def run_compute_grpo_clip_loss(*args, **kwargs): return compute_grpo_clip_loss(*args, **kwargs)
def run_compute_policy_gradient_loss(*args, **kwargs): return compute_policy_gradient_loss(*args, **kwargs)
def run_masked_mean(*args, **kwargs): return masked_mean(*args, **kwargs)
def run_sft_microbatch_train_step(*args, **kwargs): return sft_microbatch_train_step(*args, **kwargs)
def run_grpo_microbatch_train_step(*args, **kwargs): return grpo_microbatch_train_step(*args, **kwargs)
def run_masked_normalize(*args, **kwargs): return masked_normalize(*args, **kwargs)

def get_packed_sft_dataset(*args, **kwargs): return get_packed_sft_dataset_impl(*args, **kwargs)
def run_iterate_batches(*args, **kwargs): return iterate_batches_impl(*args, **kwargs)
def run_parse_mmlu_response(*args, **kwargs): return parse_mmlu_response_impl(*args, **kwargs)
def run_parse_gsm8k_response(*args, **kwargs): return parse_gsm8k_response_impl(*args, **kwargs)
def run_compute_per_instance_dpo_loss(*args, **kwargs): return compute_per_instance_dpo_loss_impl(*args, **kwargs)
