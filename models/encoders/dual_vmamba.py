import torch
import torch.nn as nn
import sys

sys.path.append('../..')
sys.path.append('..')
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torchvision import transforms
import math
import time
import numpy as np
from PIL import Image
from engine.logger import get_logger
from models.encoders.vmamba import Backbone_VSSM, CrossMambaFusionBlock, ConcatMambaFusionBlock, \
    ImprovedCrossMambaFusionBlock
from models.loss.proco import ProCoLoss
from models.loss.logitadjust import LogitAdjust
from models.encoders.Edge import GatedEdgeInjection
from models.encoders.Edge import AdaptiveEdgeInjection
from models.loss.dice import EdgeDiceLoss

logger = get_logger()


class EdgeInjection(nn.Module):
    def __init__(self, in_channels):
        super(EdgeInjection, self).__init__()
        # Sobel filters（固定不训练）
        sobel_0 = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).reshape(1, 1, 3, 3)
        sobel_90 = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).reshape(1, 1, 3, 3)
        sobel_stack = torch.cat([sobel_0, sobel_90], dim=0)
        sobel_kernel = sobel_stack.repeat(in_channels, 1, 1, 1)

        self.edge_conv = nn.Conv2d(in_channels, 2 * in_channels, kernel_size=3, padding=1, groups=in_channels,
                                   bias=False)
        self.edge_conv.weight.data = sobel_kernel
        self.edge_conv.weight.requires_grad = False

        self.reduce = nn.Conv2d(2 * in_channels, in_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        edge_feat = self.edge_conv(x)
        edge_feat = self.reduce(edge_feat)
        edge_mask = self.sigmoid(edge_feat)
        return x * edge_mask


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features_A, features_B):
        # 归一化特征
        features_A = F.normalize(features_A, dim=1)
        features_B = F.normalize(features_B, dim=1)

        # 对比损失 (NT-Xent Loss)
        similarity_matrix = torch.mm(features_A, features_B.T) / self.temperature
        labels = torch.arange(features_A.size(0)).to(features_A.device)
        loss = F.cross_entropy(similarity_matrix, labels)
        return loss


class RGBXTransformer1(nn.Module):
    def __init__(self,
                 num_classes=2,
                 temperature=1.0,  # 温度默认为1.0
                 norm_layer=nn.LayerNorm,
                 depths=[2, 2, 27, 2],  # [2,2,27,2] for vmamba small
                 dims=96,
                 pretrained=None,
                 mlp_ratio=4.0,
                 downsample_version='v1',
                 ape=False,
                 img_size=[480, 640],
                 patch_size=4,
                 drop_path_rate=0.2,
                 use_edge_modulation=True,
                 modulation_stage=2,
                 adaptive_edge_strength=True,  # 新增：自适应边缘强度
                 enable_edge_analysis=True,  # 新增：启用边缘分析
                 **kwargs):
        super().__init__()

        self.ape = ape
        self.temperature = temperature
        self.num_classes = num_classes

        self.use_edge_modulation = use_edge_modulation
        self.modulation_stage = modulation_stage
        self.adaptive_edge_strength = adaptive_edge_strength
        self.enable_edge_analysis = enable_edge_analysis

        self.contrastive_loss = ContrastiveLoss()

        # ========== 1. 边缘注入模块 ==========
        self.edge_inject_A2 = EdgeInjection(in_channels=384)
        self.edge_inject_B2 = EdgeInjection(in_channels=384)
        self.edge_inject_fused2 = EdgeInjection(in_channels=384)
        # 新增：为每个阶段添加边缘注入模块
        self.edge_inject_B0 = EdgeInjection(in_channels=96)
        self.edge_inject_B1 = EdgeInjection(in_channels=192)
        self.edge_inject_B2 = EdgeInjection(in_channels=384)
        self.edge_inject_B3 = EdgeInjection(in_channels=768)
        # self.edge_inject = EdgeInjection(in_channels=384)  # Stage 0 的维度是 96  Stage1:192  Stage2:384  Stage3:768

        # ========== 2. 自适应边缘强度参数 ==========
        if adaptive_edge_strength:
            # 每个阶段一个强度参数（可学习）
            self.edge_strength_params = nn.ParameterList([
                nn.Parameter(torch.tensor(0.5)) for _ in range(4)  # stage0-3
            ])

            # 每个注入模块的独立强度参数
            self.injection_strength_params = nn.ParameterDict({
                'A2': nn.Parameter(torch.tensor(0.5)),
                'B0': nn.Parameter(torch.tensor(0.5)),
                'B1': nn.Parameter(torch.tensor(0.5)),
                'B2': nn.Parameter(torch.tensor(0.5)),
                'B3': nn.Parameter(torch.tensor(0.5)),
                'fused2': nn.Parameter(torch.tensor(0.5)),
            })

            # 边缘重要性门控网络
            self.edge_gates = nn.ModuleList([
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(dims * (2 ** i), dims * (2 ** i) // 4, 1),
                    nn.ReLU(),
                    nn.Conv2d(dims * (2 ** i) // 4, 1, 1),
                    nn.Sigmoid()
                ) for i in range(4)
            ])

        # ========== 3. 边缘调制模块 ==========
        if use_edge_modulation:
            self.edge_modulators = nn.ModuleList()
            for i in range(4):
                in_channels = dims * (2 ** i)
                self.edge_modulators.append(
                    AdaptiveEdgeModulation(in_channels)
                )

        # 主干网络
        self.vssm = Backbone_VSSM(
            pretrained=pretrained,
            norm_layer=norm_layer,
            num_classes=num_classes,
            depths=depths,
            dims=dims,
            mlp_ratio=mlp_ratio,
            downsample_version=downsample_version,
            drop_path_rate=drop_path_rate,
        )

        self.cross_mamba = nn.ModuleList(
            ImprovedCrossMambaFusionBlock(
                hidden_dim=dims * (2 ** i),
                mlp_ratio=0.0,
                d_state=4,
            ) for i in range(4)
        )

        # 通道注意力模块，用于对融合后的通道信息进行增强。
        self.channel_attn_mamba = nn.ModuleList(
            ConcatMambaFusionBlock(
                hidden_dim=dims * (2 ** i),
                mlp_ratio=0.0,
                d_state=4,
            ) for i in range(4)
        )

        # absolute position embedding
        if self.ape:
            self.patches_resolution = [img_size[0] // patch_size, img_size[1] // patch_size]
            self.absolute_pos_embed = []
            self.absolute_pos_embed_x = []
            for i_layer in range(len(depths)):
                input_resolution = (self.patches_resolution[0] // (2 ** i_layer),
                                    self.patches_resolution[1] // (2 ** i_layer))
                dim = int(dims * (2 ** i_layer))
                absolute_pos_embed = nn.Parameter(torch.zeros(1, dim, input_resolution[0], input_resolution[1]))
                trunc_normal_(absolute_pos_embed, std=.02)
                absolute_pos_embed_x = nn.Parameter(torch.zeros(1, dim, input_resolution[0], input_resolution[1]))
                trunc_normal_(absolute_pos_embed_x, std=.02)

                self.absolute_pos_embed.append(absolute_pos_embed)
                self.absolute_pos_embed_x.append(absolute_pos_embed_x)

        # 用于记录分析数据
        self.edge_analysis_data = {
            'strength_params': [],
            'edge_gate_values': [],
            'edge_magnitudes': [],
            'modulation_effects': []
        }

    def extract_edge_maps(self, features):
        """改进的边缘图提取"""
        batch_size, channels, height, width = features.shape

        # 使用多尺度梯度
        edge_maps = []
        scales = [1, 2, 4]  # 多个尺度

        for scale in scales:
            if scale > 1:
                # 下采样
                scaled_feat = F.avg_pool2d(features, kernel_size=scale, stride=scale)
                h_scaled, w_scaled = height // scale, width // scale
            else:
                scaled_feat = features
                h_scaled, w_scaled = height, width

            # 计算梯度
            grad_x = torch.abs(scaled_feat[:, :, :, 1:] - scaled_feat[:, :, :, :-1])
            grad_y = torch.abs(scaled_feat[:, :, 1:, :] - scaled_feat[:, :, :-1, :])

            # 填充以保持尺寸
            grad_x = F.pad(grad_x, (0, 1, 0, 0))
            grad_y = F.pad(grad_y, (0, 0, 0, 1))

            # 合并梯度
            edge_map = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
            edge_map = torch.mean(edge_map, dim=1, keepdim=True)

            # 上采样回原始尺寸
            if scale > 1:
                edge_map = F.interpolate(edge_map, size=(height, width),
                                         mode='bilinear', align_corners=True)

            edge_maps.append(edge_map)

        # 融合多尺度边缘图
        fused_edge = torch.mean(torch.stack(edge_maps, dim=0), dim=0)

        # 归一化
        fused_edge = (fused_edge - fused_edge.min()) / (fused_edge.max() - fused_edge.min() + 1e-6)

        return fused_edge

    def apply_adaptive_edge_injection(self, sar_features, stage_idx):
        """应用自适应边缘注入"""
        if stage_idx == 0:
            edge_module = self.edge_inject_B0
            strength_param = self.injection_strength_params['B0']
        elif stage_idx == 1:
            edge_module = self.edge_inject_B1
            strength_param = self.injection_strength_params['B1']
        elif stage_idx == 2:
            edge_module = self.edge_inject_B2
            strength_param = self.injection_strength_params['B2']
        elif stage_idx == 3:
            edge_module = self.edge_inject_B3
            strength_param = self.injection_strength_params['B3']
        else:
            return sar_features

        # 计算边缘门控
        if self.adaptive_edge_strength:
            edge_gate = self.edge_gates[stage_idx](sar_features)

            # 应用边缘注入（带门控和自适应强度）
            edge_features = edge_module(sar_features)
            modulated_edge = edge_features * edge_gate

            # 应用自适应强度
            strength = torch.sigmoid(strength_param) * 2  # 限制在[0, 2]范围
            injected_features = sar_features + strength * modulated_edge

            # 记录分析数据
            if self.enable_edge_analysis and self.training:
                self.edge_analysis_data['strength_params'].append(strength.item())
                self.edge_analysis_data['edge_gate_values'].append(edge_gate.mean().item())
                self.edge_analysis_data['edge_magnitudes'].append(
                    torch.mean(torch.abs(modulated_edge)).item()
                )
        else:
            # 固定强度注入
            injected_features = sar_features + edge_module(sar_features)

        return injected_features

    def apply_edge_modulation(self, rgb_features, sar_features, stage_idx):
        """应用边缘调制"""
        if not self.use_edge_modulation or stage_idx != self.modulation_stage:
            return rgb_features

        # 提取边缘图
        edge_map = self.extract_edge_maps(sar_features)

        # 应用调制
        modulated_rgb = self.edge_modulators[stage_idx](rgb_features, edge_map)

        # 记录调制效果
        if self.enable_edge_analysis and self.training:
            modulation_effect = torch.mean(torch.abs(modulated_rgb - rgb_features)).item()
            self.edge_analysis_data['modulation_effects'].append(modulation_effect)

        return modulated_rgb

    def forward_features(self, x_A, x_B):
        """
        前向传播特征提取
        """
        print(f"🔧 [forward_features] Called")

        B = x_A.shape[0]
        outs_fused = []

        # 清空分析数据（每个batch开始时）
        if self.enable_edge_analysis:
            self.edge_analysis_data = {k: [] for k in self.edge_analysis_data.keys()}

        # 提取特征
        outs_A = self.vssm(x_A)
        outs_B = self.vssm(x_B)

        # ========== 边缘注入阶段 ==========
        # 对SAR特征进行多阶段边缘注入
        for i in range(4):
            outs_B[i] = self.apply_adaptive_edge_injection(outs_B[i], i)

        # 对光学特征进行边缘注入（可选）
        outs_A[2] = outs_A[2] + self.injection_strength_params['A2'] * self.edge_inject_A2(outs_A[2])

        # ========== 边缘调制阶段 ==========
        contrastive_loss = 0

        for i in range(4):
            if self.ape:
                out_A = self.absolute_pos_embed[i].to(outs_A[i].device) + outs_A[i]
                out_B = self.absolute_pos_embed_x[i].to(outs_B[i].device) + outs_B[i]
            else:
                out_A = outs_A[i]
                out_B = outs_B[i]

            # 对比损失
            contrastive_loss += self.contrastive_loss(
                out_A.view(out_A.size(0), -1),
                out_B.view(out_B.size(0), -1)
            )

            # 应用边缘调制
            modulated_A = self.apply_edge_modulation(out_A, out_B, i)
            # 在边缘调制处添加打印


            # 跨模态融合
            cross_A, cross_B = self.cross_mamba[i](
                modulated_A.permute(0, 2, 3, 1).contiguous(),
                out_B.permute(0, 2, 3, 1).contiguous()
            )

            # 通道注意力融合
            x_fuse = self.channel_attn_mamba[i](cross_A, cross_B).permute(0, 3, 1, 2).contiguous()

            # 融合后边缘增强
            if i == 2:
                fused_strength = torch.sigmoid(self.injection_strength_params['fused2'])
                x_fuse = x_fuse + fused_strength * self.edge_inject_fused2(x_fuse)

            outs_fused.append(x_fuse)

            # ========== 分析输出 ==========
            if self.enable_edge_analysis and self.training and i == self.modulation_stage:
                self._analyze_edges(out_A, out_B, modulated_A, i)

        return outs_fused, contrastive_loss

    def _analyze_edges(self, rgb_feat, sar_feat, modulated_rgb, stage_idx):
        """分析边缘效果"""
        edge_map = self.extract_edge_maps(sar_feat)

        # 计算指标
        edge_mean = edge_map.mean().item()
        edge_std = edge_map.std().item()
        edge_sparsity = (edge_map < 0.1).float().mean().item()  # 稀疏度

        # 调制前后差异
        rgb_mean = rgb_feat.mean().item()
        modulated_mean = modulated_rgb.mean().item()
        modulation_change = abs(modulated_mean - rgb_mean) / (abs(rgb_mean) + 1e-6)

        # 打印分析结果（每10个batch打印一次）
        if hasattr(self, '_analysis_counter'):
            self._analysis_counter += 1
        else:
            self._analysis_counter = 0

        if self._analysis_counter % 10 == 0:
            print(f"\n📊 [Edge Analysis - Stage {stage_idx}]")
            print(f"  Edge map - Mean: {edge_mean:.4f}, Std: {edge_std:.4f}, Sparsity: {edge_sparsity:.3f}")
            print(f"  RGB feat - Mean: {rgb_mean:.4f}")
            print(f"  Modulated RGB - Mean: {modulated_mean:.4f}, Change: {modulation_change:.2%}")

            if self.adaptive_edge_strength:
                print(f"  Strength params: {[p.item() for p in self.edge_strength_params]}")
                print(f"  Injection strengths: {[v.item() for k, v in self.injection_strength_params.items()]}")

    def get_edge_analysis_report(self):
        """获取边缘分析报告"""
        if not self.enable_edge_analysis or not self.training:
            return None

        report = {
            'avg_strength': np.mean(self.edge_analysis_data['strength_params']) if self.edge_analysis_data[
                'strength_params'] else 0,
            'avg_gate_value': np.mean(self.edge_analysis_data['edge_gate_values']) if self.edge_analysis_data[
                'edge_gate_values'] else 0,
            'avg_edge_magnitude': np.mean(self.edge_analysis_data['edge_magnitudes']) if self.edge_analysis_data[
                'edge_magnitudes'] else 0,
            'avg_modulation_effect': np.mean(self.edge_analysis_data['modulation_effects']) if self.edge_analysis_data[
                'modulation_effects'] else 0,
        }

        return report

    def forward(self, x_A, x_B,label=None):
        out, contrastive_loss = self.forward_features(x_A, x_B)  # 调用 forward_features 提取融合后的特征
        return out, contrastive_loss
        # out = self.forward_features(x_A, x_B)  # 调用 forward_features 提取融合后的特征。
        # return out


class RGBXTransformer(nn.Module):
    def __init__(self,
                 num_classes=2,
                 temperature=1.0,
                 norm_layer=nn.LayerNorm,
                 depths=[2, 2, 27, 2],
                 dims=96,
                 pretrained=None,
                 mlp_ratio=4.0,
                 downsample_version='v1',
                 ape=False,
                 img_size=[480, 640],
                 patch_size=4,
                 drop_path_rate=0.2,
                 edge_modulation=True,  # 新增：是否使用边缘调制
                 modulation_stage=2,  # 新增：在哪一阶段进行调制（默认stage2）
                 modulation_type='spatial_attention',  # 调制类型
                 use_edge_injection=True,
                 injection_stages=[0,1, 2],
                 **kwargs):
        super().__init__()

        self.ape = ape
        self.temperature = temperature
        self.num_classes = num_classes
        self.edge_modulation = edge_modulation
        self.modulation_stage = modulation_stage
        self.modulation_type = modulation_type
        self.use_edge_injection = use_edge_injection
        self.injection_stages = injection_stages

        # 边缘损失相关的权重和超参数
        self.use_edge_loss = True  # 是否使用边缘损失
        self.edge_loss_weight = 0.1  # 边缘损失权重
        self.edge_loss_type = 'dice'  # 可选：'mse', 'l1', 'dice'

        # 如果使用Dice损失
        if self.edge_loss_type == 'dice':
            self.edge_dice_loss = EdgeDiceLoss()

        print(
            f"🔧 [RGBXTransformer] Initialized with edge_modulation={edge_modulation}, modulation_stage={modulation_stage}")

        self.contrastive_loss = ContrastiveLoss()

        # 边缘注入模块（保留原有的）
        self.edge_inject_A1 = EdgeInjection(in_channels=192)
        self.edge_inject_B2 = EdgeInjection(in_channels=384)
        self.edge_inject_fused2 = EdgeInjection(in_channels=384)
        if use_edge_injection:
            self.edge_inject_A0 = EdgeInjection(in_channels=96)
            self.edge_inject_B0 = EdgeInjection(in_channels=96)
            self.edge_inject_A2 = EdgeInjection(in_channels=384)
            self.edge_inject_B1 = EdgeInjection(in_channels=192)

        # 边缘调制模块
        if edge_modulation:
            self.edge_modulation_modules = nn.ModuleList()
            for i in range(4):
                in_channels = dims * (2 ** i)
                if modulation_type == 'spatial_attention':
                    self.edge_modulation_modules.append(
                        SpatialAttentionModulation(in_channels)
                    )
                elif modulation_type == 'adaptive_weight':
                    self.edge_modulation_modules.append(
                        AdaptiveWeightModulation(in_channels)
                    )
                elif modulation_type == 'channel_attention':
                    self.edge_modulation_modules.append(
                        ChannelAttentionModulation(in_channels)
                    )
                else:
                    self.edge_modulation_modules.append(
                        SimpleModulation(in_channels)
                    )

        # 主干网络
        self.vssm = Backbone_VSSM(
            pretrained=pretrained,
            norm_layer=norm_layer,
            num_classes=num_classes,
            depths=depths,
            dims=dims,
            mlp_ratio=mlp_ratio,
            downsample_version=downsample_version,
            drop_path_rate=drop_path_rate,
        )

        # 跨模态融合模块
        self.cross_mamba = nn.ModuleList(
            ImprovedCrossMambaFusionBlock(
                hidden_dim=dims * (2 ** i),
                mlp_ratio=0.0,
                d_state=4,
            ) for i in range(4)
        )

        # 通道注意力模块
        self.channel_attn_mamba = nn.ModuleList(
            ConcatMambaFusionBlock(
                hidden_dim=dims * (2 ** i),
                mlp_ratio=0.0,
                d_state=4,
            ) for i in range(4)
        )

        # absolute position embedding
        if self.ape:
            self.patches_resolution = [img_size[0] // patch_size, img_size[1] // patch_size]
            self.absolute_pos_embed = []
            self.absolute_pos_embed_x = []
            for i_layer in range(len(depths)):
                input_resolution = (self.patches_resolution[0] // (2 ** i_layer),
                                    self.patches_resolution[1] // (2 ** i_layer))
                dim = int(dims * (2 ** i_layer))
                absolute_pos_embed = nn.Parameter(torch.zeros(1, dim, input_resolution[0], input_resolution[1]))
                trunc_normal_(absolute_pos_embed, std=.02)
                absolute_pos_embed_x = nn.Parameter(torch.zeros(1, dim, input_resolution[0], input_resolution[1]))
                trunc_normal_(absolute_pos_embed_x, std=.02)

                self.absolute_pos_embed.append(absolute_pos_embed)
                self.absolute_pos_embed_x.append(absolute_pos_embed_x)

    def compute_edge_loss(self, rgb_features, sar_features, predictions=None):
        """
        计算边缘损失
        思路：让模型预测的边缘与从SAR提取的边缘对齐
        """
        # 从SAR特征提取真实的边缘图
        sar_edge_maps = self.extract_edge_maps(sar_features)

        # 从RGB特征提取预测的边缘图
        rgb_edge_maps = self.extract_edge_maps(rgb_features)

        # 调整大小使它们匹配
        if sar_edge_maps.shape[2:] != rgb_edge_maps.shape[2:]:
            sar_edge_maps = F.interpolate(sar_edge_maps,
                                          size=rgb_edge_maps.shape[2:],
                                          mode='bilinear',
                                          align_corners=True)

        # 根据损失类型计算损失
        if self.edge_loss_type == 'mse':
            loss = F.mse_loss(rgb_edge_maps, sar_edge_maps)
        elif self.edge_loss_type == 'l1':
            loss = F.l1_loss(rgb_edge_maps, sar_edge_maps)
        elif self.edge_loss_type == 'dice':
            loss = self.edge_dice_loss(rgb_edge_maps, sar_edge_maps)

        return loss * self.edge_loss_weight

    def extract_edge_maps(self, sar_features):
        """
        从SAR特征中提取边缘图
        """
        batch_size, channels, height, width = sar_features.shape

        # 简单的梯度计算
        # x方向梯度
        grad_x = torch.abs(sar_features[:, :, :, 1:] - sar_features[:, :, :, :-1])
        grad_x = F.pad(grad_x, (0, 1, 0, 0))

        # y方向梯度
        grad_y = torch.abs(sar_features[:, :, 1:, :] - sar_features[:, :, :-1, :])
        grad_y = F.pad(grad_y, (0, 0, 0, 1))

        # 计算梯度幅度
        gradient_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)

        # 平均所有通道
        edge_map = torch.mean(gradient_magnitude, dim=1, keepdim=True)

        # 归一化
        if edge_map.max() > edge_map.min() + 1e-6:
            edge_map = (edge_map - edge_map.min()) / (edge_map.max() - edge_map.min())

        return edge_map

    def sobel_filter_x(self, x):
        """Sobel滤波器 - X方向（如果需要保留）"""
        channels = x.shape[1]

        # 创建Sobel核
        sobel_kernel = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32, device=x.device).view(1, 1, 3, 3)

        # 为每个通道复制
        sobel_kernel = sobel_kernel.repeat(channels, 1, 1, 1)

        return F.conv2d(x, sobel_kernel, padding=1, groups=channels)

    def sobel_filter_y(self, x):
        """Sobel滤波器 - Y方向（如果需要保留）"""
        channels = x.shape[1]

        sobel_kernel = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32, device=x.device).view(1, 1, 3, 3)

        sobel_kernel = sobel_kernel.repeat(channels, 1, 1, 1)

        return F.conv2d(x, sobel_kernel, padding=1, groups=channels)

    def apply_edge_modulation(self, rgb_feat, sar_feat, stage_idx):
        """
        应用边缘调制
        rgb_feat: 光学特征 [B, C, H, W]
        sar_feat: SAR特征 [B, C, H, W]
        stage_idx: 阶段索引
        返回: 调制后的光学特征
        """
        if not self.edge_modulation or stage_idx != self.modulation_stage:
            return rgb_feat

        # 从SAR特征提取边缘图
        edge_map = self.extract_edge_maps(sar_feat)

        # 调整边缘图大小以匹配光学特征
        if edge_map.shape[2:] != rgb_feat.shape[2:]:
            edge_map = F.interpolate(edge_map, size=rgb_feat.shape[2:],
                                     mode='bilinear', align_corners=True)

        # 应用调制
        modulated_rgb = self.edge_modulation_modules[stage_idx](rgb_feat, edge_map)

        # 只在训练开始时打印调试信息
        if self.training and not hasattr(self, f"_modulation_debug_{stage_idx}"):
            print(f"🔧 [Edge Modulation Stage {stage_idx}]")
            print(f"  RGB feat shape: {rgb_feat.shape}")
            print(f"  SAR feat shape: {sar_feat.shape}")
            print(f"  Edge map shape: {edge_map.shape}")
            print(f"  Edge map mean: {edge_map.mean().item():.4f}")
            print(f"  Edge map std: {edge_map.std().item():.4f}")
            setattr(self, f"_modulation_debug_{stage_idx}", True)

        return modulated_rgb

    def forward_features(self, x_A, x_B):
        """
        x_A：光学图像（RGB）
        x_B：SAR图像
        """
        B = x_A.shape[0]
        outs_fused = []

        edge_losses = []  # 存储各阶段的边缘损失

        # 提取光学和SAR特征
        outs_A = self.vssm(x_A)  # 光学特征
        outs_B = self.vssm(x_B)  # SAR特征

        # ====== 边缘注入阶段 ======
        if self.use_edge_injection:
            # 对SAR特征进行多阶段边缘注入
            if 0 in self.injection_stages:
                outs_B[0] = outs_B[0] + self.edge_inject_B0(outs_B[0])
            if 1 in self.injection_stages:
                outs_B[1] = outs_B[1] + self.edge_inject_B1(outs_B[1])
            if 2 in self.injection_stages:
                outs_B[2] = outs_B[2] + self.edge_inject_B2(outs_B[2])

            # 可选：对光学特征也进行边缘注入
            if 0 in self.injection_stages:
                outs_A[0] = outs_A[0] + self.edge_inject_A0(outs_A[0])
            if 1 in self.injection_stages:
                outs_A[1] = outs_A[1] + self.edge_inject_A1(outs_A[1])
            if 2 in self.injection_stages:
                outs_A[2] = outs_A[2] + self.edge_inject_A2(outs_A[2])


        contrastive_loss = 0

        # 遍历每个阶段的特征
        for i in range(4):
            if self.ape:
                out_A = self.absolute_pos_embed[i].to(outs_A[i].device) + outs_A[i]
                out_B = self.absolute_pos_embed_x[i].to(outs_B[i].device) + outs_B[i]
            else:
                out_A = outs_A[i]
                out_B = outs_B[i]

            # 对比损失
            contrastive_loss += self.contrastive_loss(
                out_A.view(out_A.size(0), -1),
                out_B.view(out_B.size(0), -1)
            )

            # 计算边缘损失（如果需要）
            if self.use_edge_loss:
                stage_edge_loss = self.compute_edge_loss(out_A, out_B)
                edge_losses.append(stage_edge_loss)

            # 应用边缘调制
            if self.edge_modulation and i == self.modulation_stage:
                # 从增强后的SAR特征提取边缘图
                edge_map = self.extract_edge_maps(out_B)
                # 调制光学特征
                modulated_A = self.edge_modulation_modules[i](out_A, edge_map)
            else:
                modulated_A = out_A

            # 跨模态融合
            cross_A, cross_B = self.cross_mamba[i](
                modulated_A.permute(0, 2, 3, 1).contiguous(),
                out_B.permute(0, 2, 3, 1).contiguous()
            )

            # 通道注意力融合
            x_fuse = self.channel_attn_mamba[i](cross_A, cross_B).permute(0, 3, 1, 2).contiguous()

            # 融合后边缘增强（仅对 Stage 2）
            if i == 2:
                x_fuse = x_fuse + self.edge_inject_fused2(x_fuse)

            outs_fused.append(x_fuse)

        # return outs_fused, contrastive_loss
        # return outs_fused
        # 计算总边缘损失
        total_edge_loss = torch.stack(edge_losses).mean() if edge_losses else 0
        return outs_fused, total_edge_loss, contrastive_loss

    def forward(self, x_A, x_B,label=None):
        # out, contrastive_loss = self.forward_features(x_A, x_B)
        # return out, contrastive_loss
        # out = self.forward_features(x_A, x_B)
        # return out
        out, edge_loss, contrastive_loss= self.forward_features(x_A, x_B)
        return out, edge_loss, contrastive_loss



class SpatialAttentionModulation(nn.Module):
    """空间注意力调制：使用边缘图作为空间注意力权重"""

    def __init__(self, in_channels):
        super().__init__()
        # 边缘图处理网络
        self.edge_processor = nn.Sequential(
            nn.Conv2d(1, in_channels // 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels, 3, padding=1),
            nn.Sigmoid()  # 输出[0, 1]的注意力权重
        )

        # 残差连接权重（可学习）
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, rgb_feat, edge_map):
        """
        rgb_feat: [B, C, H, W]
        edge_map: [B, 1, H, W]
        返回: 调制后的特征
        """
        # 处理边缘图得到空间注意力权重
        spatial_weights = self.edge_processor(edge_map)

        # 应用空间注意力调制
        modulated_feat = rgb_feat * spatial_weights

        # 残差连接：原始特征 + 调制特征
        output = (1 - self.alpha) * rgb_feat + self.alpha * modulated_feat

        return output


class AdaptiveWeightModulation(nn.Module):
    """自适应权重调制：学习边缘如何影响不同通道"""

    def __init__(self, in_channels):
        super().__init__()
        # 边缘特征提取
        self.edge_encoder = nn.Sequential(
            nn.Conv2d(1, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels, 3, padding=1),
        )

        # 自适应权重生成
        self.weight_generator = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, rgb_feat, edge_map):
        # 编码边缘图
        edge_features = self.edge_encoder(edge_map)

        # 拼接特征
        concat_features = torch.cat([rgb_feat, edge_features], dim=1)

        # 生成自适应权重
        adaptive_weights = self.weight_generator(concat_features)

        # 应用调制
        modulated_feat = rgb_feat * adaptive_weights + edge_features

        return modulated_feat


class ChannelAttentionModulation(nn.Module):
    """通道注意力调制：边缘信息指导通道注意力"""

    def __init__(self, in_channels):
        super().__init__()
        # 全局池化后的通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels + 1, in_channels // 4, 1),  # +1 for edge info
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, rgb_feat, edge_map):
        # 边缘图的全局信息
        edge_global = F.adaptive_avg_pool2d(edge_map, 1)

        # 拼接全局特征
        rgb_global = F.adaptive_avg_pool2d(rgb_feat, 1)
        concat_global = torch.cat([rgb_global, edge_global], dim=1)

        # 生成通道注意力权重
        channel_weights = self.channel_attention(concat_global)

        # 应用通道注意力
        modulated_feat = rgb_feat * channel_weights

        return modulated_feat


class SimpleModulation(nn.Module):
    """简单调制：边缘图直接作为调制信号"""

    def __init__(self, in_channels):
        super().__init__()
        # 将边缘图扩展到与特征相同的通道数
        self.edge_expander = nn.Conv2d(1, in_channels, 1)

    def forward(self, rgb_feat, edge_map):
        # 扩展边缘图
        expanded_edge = self.edge_expander(edge_map)

        # 简单加法调制
        modulated_feat = rgb_feat + expanded_edge

        return modulated_feat


# ==================== vssm_tiny 类 ====================

class vssm_tiny(RGBXTransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        # 设置默认的调制参数
        if 'edge_modulation' not in kwargs:
            kwargs['edge_modulation'] = True
        if 'modulation_stage' not in kwargs:
            kwargs['modulation_stage'] = 2  # 默认在stage2进行调制
        if 'modulation_type' not in kwargs:
            kwargs['modulation_type'] = 'spatial_attention'

        super(vssm_tiny, self).__init__(
            depths=[2, 2, 9, 2],
            dims=96,
            pretrained='pretrained/vmamba/vssmtiny_dp01_ckpt_epoch_292.pth',
            mlp_ratio=0.0,
            downsample_version='v1',
            drop_path_rate=0.2,
            **kwargs
        )
        print(f"🔧 [vssm_tiny] Edge modulation: {self.edge_modulation}")
        print(f"🔧 [vssm_tiny] Modulation stage: {self.modulation_stage}")
        print(f"🔧 [vssm_tiny] Modulation type: {self.modulation_type}")
        print(f"🔧 Number of parameters: {sum(p.numel() for p in self.parameters()):,}")


# ==================== 自适应边缘调制模块 ====================
class AdaptiveEdgeModulation(nn.Module):
    """自适应边缘调制模块"""

    def __init__(self, in_channels):
        super().__init__()

        # 边缘特征编码器
        self.edge_encoder = nn.Sequential(
            nn.Conv2d(1, in_channels // 4, 3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels, 3, padding=1),
        )

        # 自适应调制权重生成器
        self.modulation_weight_generator = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels, 1),
            nn.Sigmoid()
        )

        # 残差权重（可学习）
        self.residual_weight = nn.Parameter(torch.tensor(0.7))

    def forward(self, rgb_feat, edge_map):
        # 编码边缘图
        encoded_edge = self.edge_encoder(edge_map)

        # 生成调制权重
        concat_feat = torch.cat([rgb_feat, encoded_edge], dim=1)
        modulation_weights = self.modulation_weight_generator(concat_feat)

        # 应用调制
        modulated_feat = rgb_feat * (1 + modulation_weights * encoded_edge)

        # 残差连接
        residual_weight = torch.sigmoid(self.residual_weight)
        output = residual_weight * modulated_feat + (1 - residual_weight) * rgb_feat

        return output


# ==================== 边缘调制模块 ====================

class SpatialAttentionModulation(nn.Module):
    """空间注意力调制：使用边缘图作为空间注意力权重"""

    def __init__(self, in_channels):
        super().__init__()
        # 边缘图处理网络
        self.edge_processor = nn.Sequential(
            nn.Conv2d(1, in_channels // 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels, 3, padding=1),
            nn.Sigmoid()  # 输出[0, 1]的注意力权重
        )

        # 残差连接权重（可学习）
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, rgb_feat, edge_map):
        """
        rgb_feat: [B, C, H, W]
        edge_map: [B, 1, H, W]
        返回: 调制后的特征
        """
        # 处理边缘图得到空间注意力权重
        spatial_weights = self.edge_processor(edge_map)

        # 应用空间注意力调制
        modulated_feat = rgb_feat * spatial_weights

        # 残差连接：原始特征 + 调制特征
        output = (1 - self.alpha) * rgb_feat + self.alpha * modulated_feat

        return output


class AdaptiveWeightModulation(nn.Module):
    """自适应权重调制：学习边缘如何影响不同通道"""

    def __init__(self, in_channels):
        super().__init__()
        # 边缘特征提取
        self.edge_encoder = nn.Sequential(
            nn.Conv2d(1, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels, 3, padding=1),
        )

        # 自适应权重生成
        self.weight_generator = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, rgb_feat, edge_map):
        # 编码边缘图
        edge_features = self.edge_encoder(edge_map)

        # 拼接特征
        concat_features = torch.cat([rgb_feat, edge_features], dim=1)

        # 生成自适应权重
        adaptive_weights = self.weight_generator(concat_features)

        # 应用调制
        modulated_feat = rgb_feat * adaptive_weights + edge_features

        return modulated_feat


class ChannelAttentionModulation(nn.Module):
    """通道注意力调制：边缘信息指导通道注意力"""

    def __init__(self, in_channels):
        super().__init__()
        # 全局池化后的通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels + 1, in_channels // 4, 1),  # +1 for edge info
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, rgb_feat, edge_map):
        # 边缘图的全局信息
        edge_global = F.adaptive_avg_pool2d(edge_map, 1)

        # 拼接全局特征
        rgb_global = F.adaptive_avg_pool2d(rgb_feat, 1)
        concat_global = torch.cat([rgb_global, edge_global], dim=1)

        # 生成通道注意力权重
        channel_weights = self.channel_attention(concat_global)

        # 应用通道注意力
        modulated_feat = rgb_feat * channel_weights

        return modulated_feat


class SimpleModulation(nn.Module):
    """简单调制：边缘图直接作为调制信号"""

    def __init__(self, in_channels):
        super().__init__()
        # 将边缘图扩展到与特征相同的通道数
        self.edge_expander = nn.Conv2d(1, in_channels, 1)

    def forward(self, rgb_feat, edge_map):
        # 扩展边缘图
        expanded_edge = self.edge_expander(edge_map)

        # 简单加法调制
        modulated_feat = rgb_feat + expanded_edge

        return modulated_feat
# class vssm_tiny(RGBXTransformer):
#     def __init__(self, fuse_cfg=None, **kwargs):
#         super(vssm_tiny, self).__init__(
#             depths=[2, 2, 9, 2],
#             dims=96,
#             pretrained='pretrained/vmamba/vssmtiny_dp01_ckpt_epoch_292.pth',
#             mlp_ratio=0.0,
#             downsample_version='v1',
#             drop_path_rate=0.2,
#         )
#         print("Number of parameters: ", sum(p.numel() for p in self.parameters()))


class vssm_tiny1(RGBXTransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        # 设置自适应边缘参数
        if 'adaptive_edge_strength' not in kwargs:
            kwargs['adaptive_edge_strength'] = True
        if 'enable_edge_analysis' not in kwargs:
            kwargs['enable_edge_analysis'] = True  # 默认启用分析

        super(vssm_tiny, self).__init__(
            depths=[2, 2, 9, 2],
            dims=96,
            pretrained='pretrained/vmamba/vssmtiny_dp01_ckpt_epoch_292.pth',
            mlp_ratio=0.0,
            downsample_version='v1',
            drop_path_rate=0.2,
            **kwargs
        )
        print(f"🔧 [vssm_tiny] Adaptive edge strength enabled")
        print(f"🔧 [vssm_tiny] Edge analysis enabled")
        print(f"🔧 Number of parameters: {sum(p.numel() for p in self.parameters()):,}")

    def get_edge_stats(self):
        """获取边缘统计信息（用于监控）"""
        if hasattr(self, 'edge_analyzer'):
            return self.edge_analyzer.get_summary()
        return None


class vssm_tiny2(RGBXTransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        # 设置默认的调制参数
        if 'edge_modulation' not in kwargs:
            kwargs['edge_modulation'] = True
        if 'modulation_stage' not in kwargs:
            kwargs['modulation_stage'] = 2  # 默认在stage2进行调制
        if 'modulation_type' not in kwargs:
            kwargs['modulation_type'] = 'spatial_attention'

        super(vssm_tiny, self).__init__(
            depths=[2, 2, 9, 2],
            dims=96,
            pretrained='pretrained/vmamba/vssmtiny_dp01_ckpt_epoch_292.pth',
            mlp_ratio=0.0,
            downsample_version='v1',
            drop_path_rate=0.2,
            **kwargs
        )
        print(f"🔧 [vssm_tiny] Edge modulation: {self.edge_modulation}")
        print(f"🔧 [vssm_tiny] Modulation stage: {self.modulation_stage}")
        print(f"🔧 [vssm_tiny] Modulation type: {self.modulation_type}")
        print(f"🔧 Number of parameters: {sum(p.numel() for p in self.parameters()):,}")

class vssm_small(RGBXTransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        super(vssm_small, self).__init__(
            depths=[2, 2, 27, 2],
            dims=96,
            pretrained='pretrained/vmamba/vssmsmall_dp03_ckpt_epoch_238.pth',
            mlp_ratio=0.0,
            downsample_version='v1',
            drop_path_rate=0.3,
        )
        print("Number of parameters: ", sum(p.numel() for p in self.parameters()))


class vssm_base(RGBXTransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        super(vssm_base, self).__init__(
            depths=[2, 2, 27, 2],
            dims=128,
            pretrained='pretrained/vmamba/vssmbase_dp06_ckpt_epoch_241.pth',
            mlp_ratio=0.0,
            downsample_version='v1',
            drop_path_rate=0.6,
            # VMamba-B with droppath 0.5 + no ema. VMamba-B* represents for VMamba-B with droppath 0.6 + ema
        )
        print("Number of parameters: ", sum(p.numel() for p in self.parameters()))


# device = torch.device('cuda:1')
# if __name__ == "__main__":
#     print("model small")
#     x_A = torch.rand(8, 3, 256, 256).to(device)  # 模拟输入A
#     x_B = torch.rand(8, 3, 256, 256).to(device)  # 模拟输入B
#
#     model = vssm_small().to(device)
#     # out = model(x_A, x_B)
#
#     outs_fused, feature_A, feature_B = model(x_A, x_B)
#
#     for i, o in enumerate(outs_fused):
#         print(f"Stage {i} output shape: {o.shape}")
#     print(f"feature_A shape: {feature_A.shape}")
#     print(f"feature_B shape: {feature_B.shape}")

# 输出
# Number of parameters:  58020960
# Stage 0 output shape: torch.Size([8, 96, 64, 64])
# Stage 1 output shape: torch.Size([8, 192, 32, 32])
# Stage 2 output shape: torch.Size([8, 384, 16, 16])
# Stage 3 output shape: torch.Size([8, 768, 8, 8])

# 去掉融合模块后输出
# Number of parameters:  43649760
# Stage 0 output shape: torch.Size([8, 96, 64, 64])
# Stage 1 output shape: torch.Size([8, 192, 32, 32])
# Stage 2 output shape: torch.Size([8, 384, 16, 16])
# Stage 3 output shape: torch.Size([8, 768, 8, 8])

# 假设图片路径
# image_path_A = '/mnt/newdisk/zyy/data/multi/California/A/california_1.png'
# image_path_B = '/mnt/newdisk/zyy/data/multi/California/B/california_1.png'
image_path_A = '/mnt/newdisk/zyy/data/multi/Gloucester/A/gl_1.png'
image_path_B = '/mnt/newdisk/zyy/data/multi/Gloucester/B/gl_1.png'

device = torch.device('cuda:0')

if __name__ == "__main__":
    print("model tiny")

    # 加载图片
    img_A = Image.open(image_path_A)
    img_B = Image.open(image_path_B)

    # 获取图片的通道数
    channels_A = img_A.mode  # 查看通道类型（'RGB'，'L'等）
    channels_B = img_B.mode

    print(f"Image A channels: {channels_A}")
    print(f"Image B channels: {channels_B}")

    # 选择合适的转换
    if channels_A == 'RGB' and channels_B == 'RGB':
        transform = transforms.Compose([
            transforms.Resize((256, 256)),  # 调整图片大小
            transforms.ToTensor(),  # 转换为Tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化（可选）
        ])
    elif channels_A == 'L' and channels_B == 'L':
        transform = transforms.Compose([
            transforms.Resize((256, 256)),  # 调整图片大小
            transforms.Grayscale(num_output_channels=3),  # 将单通道灰度图转换为3通道（如需要）
            transforms.ToTensor(),  # 转换为Tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化（可选）
        ])
    else:
        raise ValueError("Unsupported image channels")

    # 应用预处理
    x_A = transform(img_A).unsqueeze(0).to(device)  # 扩展batch维度
    x_B = transform(img_B).unsqueeze(0).to(device)  # 扩展batch维度

    # 输出图像的尺寸
    print(f"x_A shape: {x_A.shape}")
    print(f"x_B shape: {x_B.shape}")

    model = vssm_tiny().to(device)

    # outs_fused, feature_A, feature_B = model(x_A, x_B)
    # for i, o in enumerate(outs_fused):
    #     print(f"Stage {i} output shape: {o.shape}")
    # print(f"feature_A shape: {feature_A.shape}")
    # print(f"feature_B shape: {feature_B.shape}")

    outs_fused, contrastive_loss = model(x_A, x_B)
    for i, o in enumerate(outs_fused):
        print(f"Stage {i} output shape: {o.shape}")
    # print(f"contrastive_loss: {contrastive_loss}")

# Number of parameters:  36497000
# Stage 0 output shape: torch.Size([1, 96, 64, 64])
# Stage 1 output shape: torch.Size([1, 192, 32, 32])
# Stage 2 output shape: torch.Size([1, 384, 16, 16])
# Stage 3 output shape: torch.Size([1, 768, 8, 8])
# contrastive_loss: tensor([[-16707.4841, -16706.1471]], device='cuda:0', dtype=torch.float64,
#        grad_fn=<AddBackward0>)

