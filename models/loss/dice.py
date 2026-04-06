import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss for edge detection
    适用于二值边缘图的Dice损失
    """

    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        """
        pred: 预测的边缘图 [B, 1, H, W]，值在0-1之间
        target: 真实边缘图 [B, 1, H, W]，值在0-1之间
        """
        # 展平成向量
        pred_flat = pred.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)

        # 计算交集和并集
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()

        # Dice系数
        dice = (2. * intersection + self.smooth) / (union + self.smooth)

        # Dice损失
        loss = 1 - dice

        return loss


class EdgeDiceLoss(nn.Module):
    """
    专门为边缘检测优化的Dice Loss
    增加了对边缘像素的权重
    """

    def __init__(self, smooth=1e-5, edge_weight=2.0):
        super(EdgeDiceLoss, self).__init__()
        self.smooth = smooth
        self.edge_weight = edge_weight

    def forward(self, pred, target):
        """
        pred: 预测的边缘图 [B, 1, H, W]
        target: 真实边缘图 [B, 1, H, W]
        """
        # 创建权重图，边缘区域权重更高
        weight_map = 1.0 + (self.edge_weight - 1.0) * target

        # 计算加权的交集和并集
        weighted_intersection = (pred * target * weight_map).sum()
        weighted_union = (pred * weight_map).sum() + (target * weight_map).sum()

        dice = (2. * weighted_intersection + self.smooth) / (weighted_union + self.smooth)

        return 1 - dice