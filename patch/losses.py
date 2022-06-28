import torch.nn as nn
from torch.nn import CosineSimilarity, L1Loss, MSELoss


def get_loss(config):
    if config.dist_loss_type == 'cossim':
        return CosLoss()
    # elif config.dist_loss_type == 'L2':
    #     return MseReverseLoss()
    # elif config.dist_loss_type == 'L1':
    #     return L1ReverseLoss()


class CosLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cos_sim = CosineSimilarity()

    def forward(self, emb1, emb2):
        return (self.cos_sim(emb1, emb2) + 1) / 2


# class MseReverseLoss(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         self.mse = MSELoss()
#
#     def forward(self, emb1, emb2):
#         return - self.mse(emb1, emb2)
#
#
# class L1ReverseLoss(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         self.l1_loss = L1Loss()
#
#     def forward(self, emb1, emb2):
#         return - self.l1_loss(emb1, emb2)
