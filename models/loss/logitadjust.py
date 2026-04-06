import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 通过对 logits 进行类别概率加权调整（log-scaling），缓解类别不平衡问题。
# class LogitAdjust(nn.Module):
#
#     def __init__(self, cls_num_list, tau=1, weight=None):
#         super(LogitAdjust, self).__init__()
#         cls_num_list = torch.cuda.FloatTensor(cls_num_list)
#         cls_p_list = cls_num_list / cls_num_list.sum()
#         m_list = tau * torch.log(cls_p_list)
#         self.m_list = m_list.view(1, -1)
#         self.weight = weight
#
#     def forward(self, x, target):
#         # print(f"x shape: {x.shape}")
#         # print(f"m_list shape: {self.m_list.shape}")
#         x_m = x + self.m_list
#         return F.cross_entropy(x_m, target, weight=self.weight)


class LogitAdjust(nn.Module):
    def __init__(self, cls_num_list, tau=1, weight=None):
        super(LogitAdjust, self).__init__()
        self.tau = tau

        # 计算类别概率
        cls_num_list = torch.cuda.FloatTensor(cls_num_list)
        cls_p_list = cls_num_list / cls_num_list.sum()  # 类别概率

        # 定义类属性
        self.class_priors = cls_p_list  # 类别概率
        self.m_list = tau * torch.log(cls_p_list).view(1, -1)  # 对数缩放调整值

        # 可选权重
        weight = torch.cuda.FloatTensor([1, 5])
        self.weight = weight

    # def forward(self, logits, target):
    #     # 调整 logits
    #     adjustment = self.class_priors / self.tau  # 使用类概率调整 logits
    #     adjusted_logits = logits + adjustment
    #     return adjusted_logits

    def forward(self, logits, target):
        # 调整 logits
        adjustment = (self.class_priors / self.tau).view(1, -1, 1, 1)  # 调整形状
        adjusted_logits = logits + adjustment  # 广播加法
        return adjusted_logits
