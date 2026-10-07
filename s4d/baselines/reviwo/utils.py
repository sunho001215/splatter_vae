"""ReViWo loss utilities.

Copied verbatim from ReViWo (github.com/lafmdp/ReViWo @ ac0c24958c83366dbdceebfb1db9f9111fce56a0, file ``common/utils.py``);
only the imports differ. The upstream repository states no license; see docs/BASELINES.md. Only the definitions that
the splatter4d ReViWo trainer and RL wrapper call are kept.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def create_adaptive_weight_map(input_img, reconstructed_img, high_weight=10.0, low_weight=1.0, threshold2= 10 / 255):
    # 计算差异图
    difference = torch.abs(input_img - reconstructed_img)
    
    # 根据差异生成权重图
    weight_map = torch.where(difference >= threshold2, high_weight, low_weight)
    return weight_map


class WeightedMSELoss(nn.Module):
    def __init__(self):
        super(WeightedMSELoss, self).__init__()

    def forward(self, input, target, weight_map):
        loss = (input - target) ** 2
        weighted_loss = loss * weight_map
        return torch.mean(weighted_loss)


def normalize_tensor(input: torch.Tensor, normalize_together: bool = False):
    if normalize_together:
        max = input.abs().max().detach()
        return input / max
    else:
        normalized_tensor = torch.zeros_like(input).to(input.device)
        for i in range(input.shape[0]):
            distance = torch.sum(input[i] ** 2)
            normalized_tensor[i, :] = input[i] / distance
        return normalized_tensor 


def compute_similarity(a, b, dim, way: str = "cosine-similarity", lower_bound: float = 0.0):
        simi = 0
        if way == "cosine-similarity":
            simi = F.cosine_similarity(a, b, dim=dim)
        elif way == "l2-distance":
            simi = -((a - b) ** 2).sum(dim=dim)
        elif way == "mixed":
            simi = 0.1 * F.cosine_similarity(a, b, dim=dim) - ((a - b) ** 2).sum(dim=dim)
        else:
            raise NotImplementedError
        return simi - lower_bound

