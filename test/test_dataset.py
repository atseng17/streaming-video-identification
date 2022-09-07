import pytest
from src.datasets import pdist, balanced_batch_sampler, hard_negative_pair_selector
import numpy as np
import pickle
import torch

def test_pdist():
    vectors = torch.tensor([[1.,2.],[7.,10.],[5.,5.]])
    dist_matrix = pdist(vectors).numpy()

    result = np.array([[  0., 100.,  25.],
                       [100.,   0.,  29.],
                       [ 25.,  29.,   0.]])
    assert np.array_equal(dist_matrix,result)


def test_hard_negative_pair_selector():
    with open('test/test_dataset_embeddings.pkl', 'rb') as f:
        embeddings = pickle.load(f)
    with open('test/test_dataset_target.pkl', 'rb') as f:
        target = pickle.load(f)
    with open('test/test_dataset_positive_pairs.pkl', 'rb') as f:
        positive_pairs = pickle.load(f)
    with open('test/test_dataset_negative_pairs.pkl', 'rb') as f:
        negative_pairs = pickle.load(f)
    p_pairs, npairs =  hard_negative_pair_selector().get_pairs(embeddings, target)
    assert np.array_equal(npairs,negative_pairs)





