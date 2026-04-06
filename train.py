import os.path as osp
import os
import sys
import time
import argparse
from tqdm import tqdm
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel

from dataloader.dataloader import get_train_loader
from models.builder import EncoderDecoder as segmodel
from dataloader.changeDataset import ChangeDataset
from dataloader.dataloader import ValPre
from utils.init_func import init_weight, group_weight
from utils.lr_policy import WarmUpPolyLR
from engine.engine import Engine
from engine.logger import get_logger
from utils.pyt_utils import all_reduce_tensor
from utils.pyt_utils import ensure_dir, link_file, load_model, parse_devices
from utils.visualize import print_iou, show_img
from engine.logger import get_logger
from utils.metric import hist_info, compute_score
from eval import SegEvaluator
import shutil
from models.loss.logitadjust import LogitAdjust

from tensorboardX import SummaryWriter
import numpy as np
import cv2

parser = argparse.ArgumentParser()
logger = get_logger()

os.environ['MASTER_PORT'] = '16005'

# 单GPU训练
# CUDA_VISIBLE_DEVICES="0" python train.py -d 0 -n "dataset_name"  (使用GPU0  若使用GPU1 -> CUDA_VISIBLE_DEVICES="1")
# 多GPU分布式训练
# NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node=2  --master_port 29502 train.py -p 29502 -d 0,1 -n "dataset_name"
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, inputs, targets):
        # 处理输入形状
        if inputs.dim() == 4:
            if inputs.size(1) == 2:  # 如果是二分类的logits [B, 2, H, W]
                # 对变化类取softmax概率
                probs = F.softmax(inputs, dim=1)[:, 1, :, :]  # 变化类的概率
            elif inputs.size(1) == 1:  # 如果是单通道输出 [B, 1, H, W]
                probs = torch.sigmoid(inputs.squeeze(1))
            else:
                probs = inputs.squeeze(1)  # 假设已经是概率
        else:
            probs = inputs

        probs = torch.clamp(probs, min=1e-7, max=1 - 1e-7)
        targets = targets.float()
        bce_loss = F.binary_cross_entropy(probs, targets, reduction='none')
        pt = torch.where(targets == 1, probs, 1 - probs)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)

        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class AdaptiveFocalLoss(nn.Module):
    """自适应调整alpha参数的Focal Loss"""

    def __init__(self, initial_alpha=0.25, gamma=1.0):
        super().__init__()
        self.alpha = initial_alpha
        self.gamma = gamma
        self.precision_history = []

    def update_alpha_based_on_precision(self, precision, epoch):
        """根据精度调整alpha"""
        self.precision_history.append(precision)

        if len(self.precision_history) >= 3:  # 检查最近3个batch
            avg_precision = np.mean(self.precision_history[-3:])

            # 精度过低 → 降低alpha（减少对变化类的过度关注）
            if avg_precision < 0.7 and epoch >= 40:
                self.alpha = max(self.alpha * 0.95, 0.15)  # 最低0.15
                print(f"  [调整] 精度低({avg_precision:.3f})，降低Focal Loss alpha到{self.alpha:.3f}")

            # 精度恢复 → 适当恢复alpha
            elif avg_precision > 0.8 and self.alpha < 0.25 and epoch >= 60:
                self.alpha = min(self.alpha * 1.05, 0.25)

    def forward(self, inputs, targets):
        # 处理输入形状
        if inputs.dim() == 4 and inputs.size(1) == 2:
            probs = F.softmax(inputs, dim=1)[:, 1, :, :]  # 变化类的概率
        elif inputs.dim() == 4 and inputs.size(1) == 1:
            probs = torch.sigmoid(inputs.squeeze(1))
        else:
            probs = torch.sigmoid(inputs)

        probs = torch.clamp(probs, min=1e-7, max=1 - 1e-7)
        targets = targets.float()

        bce_loss = F.binary_cross_entropy(probs, targets, reduction='none')
        pt = torch.where(targets == 1, probs, 1 - probs)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)

        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


def perturb_modality(image, modality='optical', noise_factor=0.1):   # 对输入的图像进行模态破坏（如置零或加入噪声）
    if modality == 'optical':
        # 对光学图像加噪声
        noise = torch.randn_like(image) * noise_factor
        image = image + noise  # 添加噪声
    elif modality == 'sar':
        # 对SAR图像置零
        image = torch.zeros_like(image)
    return image

with Engine(custom_parser=parser) as engine:
    args = parser.parse_args()
    print(args)
    
    dataset_name = args.dataset_name
    print("DATASET NAME::  ", dataset_name)
    if dataset_name == 'whu':
        from configs.config_whu import config
    elif dataset_name == 'gl':
        from configs.config_gl import config
    elif dataset_name == 'shuguang':
        from configs.config_shuguang import config
    elif dataset_name == 'cal':
        from configs.config_california import config
    elif dataset_name == 'gl2':
        from configs.config_gl2 import config
    else:
        raise ValueError('Not a valid dataset name')

    print("=======================================")
    print(config.tb_dir)
    print("=======================================")

    cudnn.benchmark = True
    seed = config.seed
    if engine.distributed:
        seed = engine.local_rank
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # data loader
    train_loader, train_sampler = get_train_loader(engine, ChangeDataset, config)

    if (engine.distributed and (engine.local_rank == 0)) or (not engine.distributed):
        tb_dir = config.tb_dir + '/{}'.format(time.strftime("%b%d_%d-%H-%M", time.localtime()))
        generate_tb_dir = config.tb_dir + '/tb'
        tb = SummaryWriter(log_dir=tb_dir)
        engine.link_tb(tb_dir, generate_tb_dir)



    # 实例化 LogitAdjust 损失函数
    logit_adjust = LogitAdjust(cls_num_list=[10, 2], tau=1.0)# tau 是温度调节参数
    # cal数据集cls_num_list=[10, 2] tau=1.0    gl数据集 cls_num_list=[8, 1] 0.8  1.2   tt tt2

    # 根据数据集调整Focal Loss参数
    if dataset_name == 'whu':
        focal_alpha = 0.25  # 变化类权重
        focal_gamma = 2.0
    elif dataset_name == 'gl':
        focal_alpha = 0.3  # gl数据集变化更少，增加变化类权重
        focal_gamma = 2.0
    elif dataset_name == 'cal':
        focal_alpha = 0.25
        focal_gamma = 2.0
    else:
        focal_alpha = 0.25
        focal_gamma = 1.0

    # config network and criterion

    # 加权交叉熵的权重设置，比如 [背景, 变化] 类的权重
    class_weights = torch.tensor([1.0, 2.0]).cuda()  # 变化类权重可以调高
    criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='mean')
    # 初始化Focal Loss
    focal_loss_fn = AdaptiveFocalLoss(initial_alpha=0.25)
    focal_loss_fn = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction='mean')
    # 设置损失权重（可以在这里调整）
    # test1
    focal_weight = 0.5  # Focal Loss权重
    ce_weight = 0.5  # 交叉熵损失权重
    edge_weight = 0.1  # 边缘损失权重
    # # test2
    # focal_weight = 0.3  # Focal Loss权重
    # ce_weight = 0.6  # 交叉熵损失权重
    # edge_weight = 0.1  # 边缘损失权重

    # 初始化自适应Focal Loss
    adaptive_focal_loss = AdaptiveFocalLoss(initial_alpha=0.25, gamma=1.0)


    criterion = nn.CrossEntropyLoss(reduction='mean')

    if engine.distributed:
        BatchNorm2d = nn.SyncBatchNorm
    else:
        BatchNorm2d = nn.BatchNorm2d
    
    model=segmodel(cfg=config, criterion=criterion, norm_layer=BatchNorm2d)
    
    # group weight and config optimizer
    base_lr = config.lr
    if engine.distributed:
        base_lr = config.lr
    
    params_list = []
    params_list = group_weight(params_list, model, BatchNorm2d, base_lr)
    
    if config.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params_list, lr=base_lr, betas=(0.9, 0.999), weight_decay=config.weight_decay)
    elif config.optimizer == 'SGDM':
        optimizer = torch.optim.SGD(params_list, lr=base_lr, momentum=config.momentum, weight_decay=config.weight_decay)
    else:
        raise NotImplementedError

    # config lr policy
    total_iteration = config.nepochs * config.niters_per_epoch
    lr_policy = WarmUpPolyLR(base_lr, config.lr_power, total_iteration, config.niters_per_epoch * config.warm_up_epoch)

    if engine.distributed:
        logger.info('.............distributed training.............')
        if torch.cuda.is_available():
            model.cuda()
            model = DistributedDataParallel(model, device_ids=[engine.local_rank], 
                                            output_device=engine.local_rank, find_unused_parameters=True)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

    engine.register_state(dataloader=train_loader, model=model,
                          optimizer=optimizer)
    if engine.continue_state_object:
        engine.restore_checkpoint()

    optimizer.zero_grad()
    model.train()
    logger.info('begin trainning:')

    # # 打印损失权重信息
    # print("=" * 60)
    # print("损失函数配置:")
    # print(f"  Focal Loss: alpha={focal_alpha}, gamma={focal_gamma}")
    # print(f"  损失权重: Focal={focal_weight}, CE={ce_weight}, Edge={edge_weight}")
    # print("=" * 60)
    
    # Initialize the evaluation dataset and evaluator
    val_setting = {'root': config.root_folder,
                    'A_format': config.A_eval_format,
                    'B_format': config.B_eval_format,
                    'gt_format': config.gt_eval_format,
                    'class_names': config.class_names}

    val_pre = ValPre()
    val_dataset = ChangeDataset(val_setting, 'val', val_pre)

    best_mean_iou = 0.0  # Track the best mean IoU for model saving
    best_epoch = 100000  # Track the epoch with the best mean IoU for model saving

    for epoch in range(engine.state.epoch, config.nepochs+1):
        if engine.distributed:
            train_sampler.set_epoch(epoch)
        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
        pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout,
                    bar_format=bar_format)
        dataloader = iter(train_loader)

        sum_loss = 0
        sum_focal_loss = 0
        sum_ce_loss = 0
        sum_edge_loss = 0
        sum_precision_loss = 0
        sum_recall_loss = 0

        # 添加epoch级别的监控
        epoch_precision_list = []
        epoch_recall_list = []

        for idx in pbar:
            engine.update_iteration(epoch, idx)
            try:
                minibatch = next(dataloader)
            except StopIteration:
                break
            As = minibatch['A']     # 光学图像
            Bs = minibatch['B']     # SAR图像
            gts = minibatch['gt']   # 真实标签

            As = As.cuda(non_blocking=True)
            Bs = Bs.cuda(non_blocking=True)
            gts = gts.cuda(non_blocking=True)

            # 选择性破坏模态
            # As_perturbed = perturb_modality(As, modality='optical', noise_factor=0.1)
            # Bs_perturbed = perturb_modality(Bs, modality='sar')

            # 曙光数据集 较为特殊,前As:SAR 后Bs:光学
            # As_perturbed = perturb_modality(As, modality='sar')
            # Bs_perturbed = perturb_modality(Bs, modality='optical', noise_factor=0.1)

            # 破坏后模型推理
            # logits = model(As_perturbed, Bs)  # 使用破坏后的光学图像与原始SAR图像
            # logits = model(As, Bs_perturbed)  # 使用原始光学图像与破坏后的SAR图像

            aux_rate = 0.2
            # loss = model(As, Bs, gts)

            # 模型输出 logits
            # logits = model(As, Bs)
            #
            # # 调整 logits
            # adjusted_logits = logit_adjust(logits, gts)
            #
            # # 计算损失
            # loss = criterion(adjusted_logits, gts)


            # 模型前向传播
            # 根据您的模型返回，可能需要调整这里
            outputs = model(As, Bs)  # 假设模型返回logits
            # 如果模型返回多个输出（如特征和边缘损失）
            if isinstance(outputs, tuple) and len(outputs) == 2:
                logits, edge_loss_value = outputs
            elif isinstance(outputs, tuple) and len(outputs) > 2:
                # 假设第一个是logits，最后一个是edge_loss
                logits = outputs[0]
                edge_loss_value = outputs[-1] if len(outputs) > 2 else 0
            else:
                logits = outputs
                edge_loss_value = 0

            # ============== 处理logits和gts的形状 ==============
            # 确保logits是2D或4D格式
            if logits.dim() == 4:
                if logits.size(1) == 2:  # [B, 2, H, W] - 这是您的模型输出格式
                    # 取变化类的logits（第二个通道）
                    change_logits = logits[:, 1, :, :]  # [B, H, W]
                    # 计算变化类的概率
                    pred_probs = torch.sigmoid(change_logits)
                    # 二值化预测
                    pred_binary = (pred_probs > 0.5).float()  # [B, H, W]

                    # 同时保留完整的logits用于损失计算
                    full_logits = logits  # [B, 2, H, W]

                elif logits.size(1) == 1:  # [B, 1, H, W]
                    pred_probs = torch.sigmoid(logits.squeeze(1))  # [B, H, W]
                    pred_binary = (pred_probs > 0.5).float()
                    full_logits = logits
                else:
                    print(f"Warning: Unexpected logits shape: {logits.shape}")
                    # 尝试取最后一个通道
                    change_logits = logits[:, -1, :, :]
                    pred_probs = torch.sigmoid(change_logits)
                    pred_binary = (pred_probs > 0.5).float()
                    full_logits = logits
            else:
                print(f"Warning: logits has unexpected dimensions: {logits.dim()}")
                # 尝试处理
                pred_probs = torch.sigmoid(logits)
                pred_binary = (pred_probs > 0.5).float()
                full_logits = logits

            # 处理gts的形状
            if gts.dim() == 4:
                if gts.size(1) == 1:  # [B, 1, H, W]
                    gts = gts.squeeze(1)  # [B, H, W]
                elif gts.size(1) == 2:  # [B, 2, H, W] one-hot
                    gts = gts[:, 1, :, :]  # 取变化类 [B, H, W]
                else:
                    gts = gts[:, 0, :, :]  # 取第一个通道 [B, H, W]
            elif gts.dim() == 3:  # [B, H, W]，正确格式
                pass  # 保持原样
            elif gts.dim() == 2:  # [H, W] 单张图
                gts = gts.unsqueeze(0)  # [1, H, W]
            else:
                print(f"Warning: Unexpected gts shape: {gts.shape}")
                # 尝试重塑为[B, H, W]
                if gts.numel() == pred_binary.numel():
                    gts = gts.view_as(pred_binary)

            # 确保gts是0/1二值
            if gts.max() > 1 or gts.min() < 0:
                # 可能是多值，需要二值化
                gts = (gts > 0.5).float()

            # 确保pred_probs是[B, H, W]格式
            if pred_binary.shape[1:] != gts.shape[1:]:
                # 尺寸不匹配，调整pred_binary和pred_probs
                target_h, target_w = gts.shape[1], gts.shape[2]
                current_h, current_w = pred_binary.shape[1], pred_binary.shape[2]

                if current_h != target_h or current_w != target_w:
                    print(f"Adjusting size: from ({current_h}, {current_w}) to ({target_h}, {target_w})")

                    # 调整pred_binary
                    pred_binary = pred_binary.unsqueeze(1)  # [B, 1, H, W]
                    pred_binary = F.interpolate(pred_binary,
                                                size=(target_h, target_w),
                                                mode='nearest')
                    pred_binary = pred_binary.squeeze(1)  # [B, H, W]

                    # 调整pred_probs
                    pred_probs = pred_probs.unsqueeze(1)
                    pred_probs = F.interpolate(pred_probs,
                                               size=(target_h, target_w),
                                               mode='bilinear',
                                               align_corners=True)
                    pred_probs = pred_probs.squeeze(1)

                    # 调整full_logits（如果需要）
                    if full_logits.dim() == 4:
                        full_logits = F.interpolate(full_logits,
                                                    size=(target_h, target_w),
                                                    mode='bilinear',
                                                    align_corners=True)

            # 调整 logits
            adjusted_logits = logit_adjust(logits, gts)

            # ============== 计算多种损失 ==============
            # 1. Focal Loss
            focal_loss_value = focal_loss_fn(adjusted_logits, gts)
            # 2. 传统的交叉熵损失
            ce_loss_value = criterion(adjusted_logits, gts)
            # 3. 边缘损失（从模型返回）
            if isinstance(edge_loss_value, torch.Tensor):
                edge_loss_value_scaled = edge_loss_value
            else:
                edge_loss_value_scaled = torch.tensor(0.0).cuda()
            # 4. 总损失（加权组合）
            total_loss = (focal_weight * focal_loss_value +
                          ce_weight * ce_loss_value +
                          edge_weight * edge_loss_value)
            loss = total_loss
            # ============== 计算当前batch的性能指标 ==============
            with torch.no_grad():

                if pred_binary.shape != gts.shape:
                    print(f"Error: Final shape mismatch")
                    print(f"  pred_binary: {pred_binary.shape}")
                    print(f"  gts: {gts.shape}")
                    # 尝试最后的修复
                    if pred_binary.dim() == 4 and pred_binary.shape[1] > 1:
                        # 如果是[B, C, H, W]，取第一个通道
                        pred_binary = pred_binary[:, 0, :, :]
                    if pred_probs.dim() == 4 and pred_probs.shape[1] > 1:
                        pred_probs = pred_probs[:, 0, :, :]

                # 再次检查形状
                if pred_binary.shape == gts.shape:
                    # 计算TP, FP, FN
                    tp = ((pred_binary == 1) & (gts == 1)).sum().item()
                    fp = ((pred_binary == 1) & (gts == 0)).sum().item()
                    fn = ((pred_binary == 0) & (gts == 1)).sum().item()

                    current_precision = tp / (tp + fp + 1e-7)
                    current_recall = tp / (tp + fn + 1e-7) if (tp + fn) > 0 else 0
                else:
                    # 如果形状仍然不匹配，使用默认值
                    print(f"Warning: Cannot compute metrics due to shape mismatch")
                    print(f"  pred_binary: {pred_binary.shape}")
                    print(f"  gts: {gts.shape}")
                    current_precision = 0.5
                    current_recall = 0.5

                # 记录到epoch列表
                epoch_precision_list.append(current_precision)
                epoch_recall_list.append(current_recall)

            loss = total_loss

            # reduce the whole loss over multi-gpu
            # if engine.distributed:
            #     reduce_loss = all_reduce_tensor(loss, world_size=engine.world_size)
            if engine.distributed:
                reduce_total_loss = all_reduce_tensor(total_loss, world_size=engine.world_size)
                reduce_focal_loss = all_reduce_tensor(focal_loss_value, world_size=engine.world_size)
                reduce_ce_loss = all_reduce_tensor(ce_loss_value, world_size=engine.world_size)
                if isinstance(edge_loss_value, torch.Tensor):
                    reduce_edge_loss = all_reduce_tensor(edge_loss_value_scaled, world_size=engine.world_size)
                else:
                    reduce_edge_loss = torch.tensor(0.0).cuda()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            current_idx = (epoch- 1) * config.niters_per_epoch + idx 
            lr = lr_policy.get_lr(current_idx)

            for i in range(len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = lr


            if engine.distributed:
                if dist.get_rank() == 0:
                    sum_loss += reduce_total_loss.item()
                    sum_focal_loss += reduce_focal_loss.item()
                    sum_ce_loss += reduce_ce_loss.item()
                    sum_edge_loss += reduce_edge_loss.item()


                    # 打印当前iteration的损失
                    print_str = f'Epoch {epoch}/{config.nepochs}' \
                                + f' Iter {idx + 1}/{config.niters_per_epoch}' \
                                + f' lr={lr:.4e}' \
                                + f' loss={reduce_total_loss.item():.4f}' \
                                + f' (focal:{reduce_focal_loss.item():.4f}' \
                                + f' ce:{reduce_ce_loss.item():.4f}' \
                                + f' edge:{reduce_edge_loss.item():.4f})'
                    pbar.set_description(print_str, refresh=False)
            else:
                sum_loss += total_loss.item()
                sum_focal_loss += focal_loss_value.item()
                sum_ce_loss += ce_loss_value.item()
                if isinstance(edge_loss_value, torch.Tensor):
                    sum_edge_loss += edge_loss_value_scaled.item()

                # 打印当前iteration的损失
                print_str = f'Epoch {epoch}/{config.nepochs}' \
                            + f' Iter {idx + 1}/{config.niters_per_epoch}' \
                            + f' lr={lr:.4e}' \
                            + f' loss={total_loss.item():.4f}' \
                            + f' (focal:{focal_loss_value.item():.4f}' \
                            + f' ce:{ce_loss_value.item():.4f}' \
                            + f' edge:{edge_loss_value_scaled.item() if isinstance(edge_loss_value, torch.Tensor) else 0:.4f})'
                pbar.set_description(print_str, refresh=False)

            del total_loss, focal_loss_value, ce_loss_value, logits


        if (engine.distributed and (engine.local_rank == 0)) or (not engine.distributed):
            tb.add_scalar('train_loss', sum_loss / len(pbar), epoch)

        if (epoch >= config.checkpoint_start_epoch) and (epoch % config.checkpoint_step == 0) or (epoch == config.nepochs):
            if engine.distributed and (engine.local_rank == 0):
                engine.save_and_link_checkpoint(config.checkpoint_dir,
                                                config.log_dir,
                                                config.log_dir_link)
            elif not engine.distributed:
                engine.save_and_link_checkpoint(config.checkpoint_dir,
                                                config.log_dir,
                                                config.log_dir_link)
        
        # devices_val = [engine.local_rank] if engine.distributed else [0]
        torch.cuda.empty_cache()
        if engine.distributed:
            if dist.get_rank() == 0:
                # only test on rank 0, otherwise there would be some synchronization problems
                # evaluation to decide whether to save the model
                if (epoch >= config.checkpoint_start_epoch) and (epoch - config.checkpoint_start_epoch) % config.checkpoint_step == 0:
                    model.eval() 
                    with torch.no_grad():
                        all_dev = parse_devices(args.devices)
                        # network = segmodel(cfg=config, criterion=None, norm_layer=nn.BatchNorm2d).cuda(all_dev[0])
                        segmentor = SegEvaluator(dataset=val_dataset, class_num=config.num_classes,
                                                norm_mean=config.norm_mean, norm_std=config.norm_std,
                                                network=model, multi_scales=config.eval_scale_array,
                                                is_flip=config.eval_flip, devices=[model.device],
                                                verbose=False, config=config
                                                )
                        _, mean_IoU = segmentor.run(config.checkpoint_dir, str(epoch), config.val_log_file,
                                    config.link_val_log_file)
                        print('mean_IoU:', mean_IoU)

                        # Add saving of prediction result here
                        if config.save_pred_images:  # You can define a flag in your config to control this behavior
                            save_pred_path = os.path.join(config.checkpoint_dir, f"pred_images_epoch_{epoch}")
                            ensure_dir(save_pred_path)  # Ensure directory exists

                            # Assuming SegEvaluator has a method to return predictions
                            pred = segmentor.pred  # Your segmentor should return predictions as numpy arrays
                            fn = f"pred_epoch_{epoch}.png"

                            # Save the image
                            pred_image = (pred * 255).astype(np.uint8)  # Scale to 0-255 for saving
                            cv2.imwrite(os.path.join(save_pred_path, fn), pred_image)

                        # Determine if the model performance improved
                        if mean_IoU > best_mean_iou:
                            # If the model improves, remove the saved checkpoint for this epoch
                            checkpoint_path = os.path.join(config.checkpoint_dir, f'epoch-{best_epoch}.pth')
                            if os.path.exists(checkpoint_path):
                                os.remove(checkpoint_path)
                            best_epoch = epoch
                            best_mean_iou = mean_IoU
                        else:
                            # If the model does not improve, remove the saved checkpoint for this epoch
                            checkpoint_path = os.path.join(config.checkpoint_dir, f'epoch-{epoch}.pth')
                            if os.path.exists(checkpoint_path):
                                os.remove(checkpoint_path)
                        
                    model.train()
        else:
            if (epoch >= config.checkpoint_start_epoch) and (epoch - config.checkpoint_start_epoch) % config.checkpoint_step == 0:
                model.eval() 
                with torch.no_grad():
                    devices_val = [engine.local_rank] if engine.distributed else [0]
                    # segmentor = SegEvaluator(dataset=val_dataset, class_num=config.num_classes,
                    #                         norm_mean=config.norm_mean, norm_std=config.norm_std,
                    #                         network=model, multi_scales=config.eval_scale_array,
                    #                         is_flip=config.eval_flip, devices=[1,2,3],
                    #                         verbose=False,
                    #                         )
                    segmentor = SegEvaluator(dataset=val_dataset, class_num=config.num_classes,
                                             norm_mean=config.norm_mean, norm_std=config.norm_std,
                                             network=model, multi_scales=config.eval_scale_array,
                                             is_flip=config.eval_flip, devices= [0],
                                             verbose=False, config=config
                                             )
                    _, mean_IoU = segmentor.run(config.checkpoint_dir, str(epoch), config.val_log_file,
                                config.link_val_log_file)
                    print('mean_IoU:', mean_IoU)

                    # Add saving of prediction result here
                    if config.save_pred_images:  # You can define a flag in your config to control this behavior
                        save_pred_path = os.path.join(config.checkpoint_dir, f"pred_images_epoch_{epoch}")
                        ensure_dir(save_pred_path)  # Ensure directory exists

                        # Assuming SegEvaluator has a method to return predictions
                        # pred = segmentor.pred  # Your segmentor should return predictions as numpy arrays
                        pred = segmentor.predictions
                        fn = f"pred_epoch_{epoch}.png"

                        pred = np.array(segmentor.predictions)  # 确保 pred 是 NumPy 数组
                        print(pred.shape)  # 打印 pred 的形状


                        # Save the image
                        # 去掉批次维度，将 shape 从 (1, 875, 500) 转换为 (875, 500)
                        pred_image = pred.squeeze(0)  # 去掉第一维 (batch dimension)
                        pred_image = (pred_image * 255).astype(np.uint8)  # Scale to 0-255 for saving
                        print(pred_image.shape)  # 输出 (875, 500)
                        cv2.imwrite(os.path.join(save_pred_path, fn), pred_image)

                    # Determine if the model performance improved
                    if mean_IoU > best_mean_iou:
                        # If the model improves, remove the saved checkpoint for this epoch
                        checkpoint_path = os.path.join(config.checkpoint_dir, f'epoch-{best_epoch}.pth')
                        if os.path.exists(checkpoint_path):
                            os.remove(checkpoint_path)
                        best_epoch = epoch
                        best_mean_iou = mean_IoU
                    else:
                        # If the model does not improve, remove the saved checkpoint for this epoch
                        checkpoint_path = os.path.join(config.checkpoint_dir, f'epoch-{epoch}.pth')
                        if os.path.exists(checkpoint_path):
                            os.remove(checkpoint_path)
                model.train()