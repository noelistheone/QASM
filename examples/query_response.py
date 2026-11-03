"""Demonstrate the shared-state quadratic readout on a synthetic example.

This example uses no dataset or trained weights and does not measure accuracy.
"""
from pathlib import Path
from types import SimpleNamespace
import json
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.controlled_query_learning import current_response
from src.response_learning import angular_weights


def main():
    torch.manual_seed(7)
    torch.set_num_threads(2)
    # Two objects, a 4x4 mean field, and three shared spatial factors per object.
    mean = torch.randn(2, 1, 4, 4)
    factors = torch.randn(2, 3, 16) * 0.1
    # Each query operator has the same field layout as the scattering readout.
    matrices = [torch.randn(2400, 16) * 0.02 for _ in range(3)]
    physical = SimpleNamespace(matrices=matrices, weights=angular_weights("cpu"))
    response = current_response(mean, factors, physical)
    mean_only = current_response(mean, torch.zeros_like(factors), physical)
    assert response.shape == (2, 15)
    assert torch.isfinite(response).all()
    assert torch.all(response >= mean_only - 1e-6)
    print(json.dumps({"synthetic_example": True, "objects": 2, "queries": 15,
                      "rank": 3, "minimum_response": float(response.min()),
                      "minimum_factor_contribution": float((response - mean_only).min())}, indent=2))


if __name__ == "__main__":
    main()
