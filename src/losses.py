import torch
import torch.nn as nn
import torch.nn.functional as F


class OnlineContrastiveLoss(nn.Module):
    """Online Contrastive loss. Similar to ConstrativeLoss, but it selects hard positive (positives that are far apart)
    and hard negative pairs (negatives that are close) and computes the loss only for these pairs. Often yields
    better performances than  ConstrativeLoss.

    Args:
        margin (float): Negative samples should have a distance of at least the margin value.
        pair_selector (torch float tensor): output 1 from siamese model
        embeddings, target (torch float tensor): output 2 from siamese model
    Returns:
        loss_contrastive (torch float tensor): loss
    """

    def __init__(self, margin, pair_selector):
        super(OnlineContrastiveLoss, self).__init__()
        self.margin = margin
        self.pair_selector = pair_selector

    def forward(self, embeddings, target):
        positive_pairs, negative_pairs = self.pair_selector.get_pairs(
            embeddings, target
        )
        if embeddings.is_cuda:
            positive_pairs = positive_pairs.cuda()
            negative_pairs = negative_pairs.cuda()
        positive_loss = (
            (embeddings[positive_pairs[:, 0]] -
             embeddings[positive_pairs[:, 1]])
            .pow(2)
            .sum(1)
        )
        negative_loss = F.relu(
            self.margin
            - (embeddings[negative_pairs[:, 0]] -
               embeddings[negative_pairs[:, 1]])
            .pow(2)
            .sum(1)
            .sqrt()
        ).pow(2)
        loss = torch.cat([positive_loss, negative_loss], dim=0)

        return loss.mean()
