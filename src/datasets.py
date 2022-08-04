import sys
import random
import logging
from itertools import combinations

import numpy as np
from PIL import Image
import torch
import torchvision
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler

logger = logging.getLogger(__name__)
streamHandler = logging.StreamHandler(sys.stdout)
logger.addHandler(streamHandler)
file_handler = logging.FileHandler("log.txt")
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s;%(levelname)s;%(message)s", "%Y-%m-%d %H:%M:%S"
)
file_handler.setFormatter(formatter)


def seed_worker(worker_id):
    """function for seeding workers for torch dataloader
    Args:
        worker_id (int): seed id
    Returns:
        None
    """
    np.random.seed(worker_id)
    random.seed(worker_id)


class BalancedBatchSampler(BatchSampler):
    """Batch Sampler that forces the dataloader to load equal n samples from n classes.
    Args:
        labels (list): a list of all lables in the dataset
        n_classes (int): number of classes
        n_samples (int): number of samples for each class
    Returns:
        None
    """

    def __init__(self, labels, n_classes, n_samples):
        self.labels = labels
        self.labels_set = list(set(self.labels))
        self.label_to_indices = {
            label: np.where(np.array(self.labels) == label)[0]
            for label in self.labels_set
        }
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(
                self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(
                    self.label_to_indices[class_][
                        self.used_label_indices_count[
                            class_
                        ]: self.used_label_indices_count[class_]
                        + self.n_samples
                    ]
                )
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(
                    self.label_to_indices[class_]
                ):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size


class PairSelector:
    """Template for pair selection for siamese net
    Args:
        embeddings (torch float tensor):
        target (torch float tensor): output 2 from siamese model
    Returns:
        None, expecting a tensor list of positive sample pair indices and a tensor list of negative sample pair indices
    """

    def __init__(self):
        pass

    def get_pairs(self, embeddings, labels):
        raise NotImplementedError


def pdist(vectors):
    """Pairwise distances for a given tensor list
    Args:
        vectors (torch float tensor): tensor with shape n samples by m dims
    Returns:
        distance_matrix (torch float tensor): a n by n matrix
    """
    distance_matrix = (
        -2 * vectors.mm(torch.t(vectors))
        + vectors.pow(2).sum(dim=1).view(1, -1)
        + vectors.pow(2).sum(dim=1).view(-1, 1)
    )
    return distance_matrix


class HardNegativePairSelector(PairSelector):
    """A data sampling class for hard negative sampling.
    """

    def __init__(self, cpu=True):
        super(HardNegativePairSelector, self).__init__()
        self.cpu = cpu

    def get_pairs(self, embeddings, labels):
        """A selector that only chooses n negative pairs with smallest distance to n positive pairs
        Args:
            embeddings (torch float tensor): output from siamese backbone
            labels (torch int tensor): label corresponding for each embedding vector
        Returns:
            positive_pairs (torch float tensor): a tensor list of positive sample pair indices
            top_negative_pairs (torch float tensor): a tensor list of negative sample pair indices
        """
        if self.cpu:
            embeddings = embeddings.cpu()
        distance_matrix = pdist(embeddings)

        labels = labels.cpu().data.numpy()
        all_pairs = np.array(list(combinations(range(len(labels)), 2)))
        all_pairs = torch.LongTensor(all_pairs)
        positive_pairs = all_pairs[
            (labels[all_pairs[:, 0]] == labels[all_pairs[:, 1]]).nonzero()
        ]
        negative_pairs = all_pairs[
            (labels[all_pairs[:, 0]] != labels[all_pairs[:, 1]]).nonzero()
        ]

        negative_distances = distance_matrix[negative_pairs[:,
                                                            0], negative_pairs[:, 1]]
        negative_distances = negative_distances.cpu().data.numpy()
        top_negatives = np.argpartition(negative_distances, len(positive_pairs))[
            : len(positive_pairs)
        ]
        top_negative_pairs = negative_pairs[torch.LongTensor(top_negatives)]

        return positive_pairs, top_negative_pairs


class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    def __getitem__(self, index):
        """A selector that only chooses n negative pairs with smallest distance to n positive pairs
        Args:
            index (int): index for retrieving 
        Returns:
            tuple_with_path (tuple): a tuple with sample (torch.tensor) and path(str)
        """
        original_tuple = super(ImageFolderWithPaths,
                               self).__getitem__(index)
        path = self.imgs[index][0]
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path
