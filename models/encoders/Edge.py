import torch.nn.functional as F
import torch.nn as nn
import torch

class GatedEdgeInjection(nn.Module):
    """门控边缘注入 - 学习何时应该注入边缘信息"""

    def __init__(self, in_channels):
        super().__init__()
        # self.in_channels = in_channels
        self.in_channels = in_channels + in_channels//4


        # 边缘提取分支
        self.edge_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels // 4, 3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
        )

        # 门控分支
        self.gate_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels // 2, 1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels, 1),
            nn.Sigmoid()  # 输出0-1的门控权重
        )

        # 输出投影
        self.output_conv = nn.Conv2d(in_channels // 4, in_channels, 1)

    def forward(self, x, edge_map=None):
        B, C, H, W = x.shape

        # 如果没有提供边缘图，从特征中提取
        if edge_map is None:
            edge_map = self.extract_edge_from_feature(x)

        # 提取边缘特征
        edge_feat = self.edge_conv(x)

        # 计算门控权重
        x_flat = F.adaptive_avg_pool2d(x, (1, 1))
        edge_flat = F.adaptive_avg_pool2d(edge_feat, (1, 1))
        gate_input = torch.cat([x_flat, edge_flat], dim=1)
        gate_input = F.interpolate(gate_input, size=(H, W), mode='bilinear', align_corners=False)

        gate_weight = self.gate_conv(gate_input)

        # 应用门控的边缘注入
        edge_enhanced = self.output_conv(edge_feat)
        output = x + gate_weight * edge_enhanced

        return output

    def extract_edge_from_feature(self, x):
        """从特征中提取边缘信息"""
        # 使用Sobel算子
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=torch.float32, device=x.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=torch.float32, device=x.device).view(1, 1, 3, 3)

        edges = []
        for i in range(x.shape[1]):
            channel = x[:, i:i + 1, :, :]
            edge_x = F.conv2d(channel, sobel_x, padding=1)
            edge_y = F.conv2d(channel, sobel_y, padding=1)
            edge_mag = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-6)
            edges.append(edge_mag)

        return torch.mean(torch.stack(edges, dim=1), dim=1, keepdim=True)


class AdaptiveEdgeInjection(nn.Module):
    """自适应边缘注入 - 动态调整注入强度"""

    def __init__(self, in_channels):
        super().__init__()
        # self.in_channels = in_channels
        self.in_channels = in_channels + in_channels // 4

        # 边缘特征提取
        self.edge_extractor = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.GroupNorm(8, in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels // 4, 3, padding=1),
            nn.GroupNorm(4, in_channels // 4),
            nn.ReLU(inplace=True),
        )

        # 自适应权重学习
        self.weight_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels + in_channels // 4, in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 2, in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 4, 1),
            nn.Sigmoid()  # 输出0-1的权重
        )

        # 特征融合
        self.fusion = nn.Conv2d(in_channels // 4, in_channels, 1)

    def forward(self, x, edge_map=None):
        B, C, H, W = x.shape

        # 提取边缘特征
        edge_feat = self.edge_extractor(x)

        # 计算自适应权重
        x_global = F.adaptive_avg_pool2d(x, (1, 1)).view(B, -1)
        edge_global = F.adaptive_avg_pool2d(edge_feat, (1, 1)).view(B, -1)

        weight_input = torch.cat([x_global, edge_global], dim=1)
        adaptive_weight = self.weight_predictor(weight_input).view(B, 1, 1, 1)

        # 应用自适应权重的边缘注入
        edge_enhanced = self.fusion(edge_feat)
        output = x + adaptive_weight * edge_enhanced

        return output