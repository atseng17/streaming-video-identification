import pytest
from src.losses import online_contrastive_loss
from src.datasets import hard_negative_pair_selector
import pickle
import torch

def test_online_contrastive_loss():
    with open('test/test_loss_labels.pkl', 'rb') as f:
        labels = pickle.load(f)
    with open('test/test_loss_output.pkl', 'rb') as f:
        output = pickle.load(f)

    margin = 1.
    criterion = online_contrastive_loss(margin, hard_negative_pair_selector())

    assert criterion(output, labels) == pytest.approx(torch.tensor(0.3384),abs=1e-4)

