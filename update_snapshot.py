import numpy as np
import torch
import torch.nn.functional as F
import sys
import os

project_dir = "/data1/ikun/1￥￥￥/LCPU/assignment5-alignment"
sys.path.insert(0, project_dir)
from tests.conftest import NumpySnapshot, _canonicalize_array
from cs336_alignment.alignment import get_response_log_probs
from transformers import AutoModelForCausalLM

def main():
    model_id = "Qwen/Qwen2.5-Math-1.5B"
    print(f"Loading model {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(model_id)
    
    # These are the exact input parameters used in tests/test_sft.py::test_get_response_log_probs
    input_ids = torch.tensor([[42, 67, 76, 14, 26, 35, 20, 24, 50, 13],
            [78, 14, 10, 54, 31, 72, 15, 95, 67,  6]])
    labels = torch.tensor([[67, 76, 14, 26, 35, 20, 24, 50, 13,  0],
            [14, 10, 54, 31, 72, 15, 95, 67,  6,  0]])

    print("Running forward pass and computing log probs...")
    output = get_response_log_probs(model, input_ids, labels, return_token_entropy=True)
    
    print("Normalizing tensors to numpy arrays...")
    output_dict = {k: _canonicalize_array(v) for k, v in output.items()}

    snapshot_path = "/data1/ikun/1￥￥￥/LCPU/assignment5-alignment/tests/_snapshots/test_get_response_log_probs.npz"
    print(f"Saving to {snapshot_path}...")
    np.savez(snapshot_path, **output_dict)
    print("Snapshot updated successfully! You can now run `pytest tests/test_sft.py::test_get_response_log_probs`")

if __name__ == "__main__":
    main()
