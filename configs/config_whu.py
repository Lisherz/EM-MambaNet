import os
import os.path as osp
import sys
import time
import numpy as np
from easydict import EasyDict as edict
import argparse

C = edict()
config = C
cfg = C

C.seed = 3407

# os.popen('pwd').read()：执行 shell 命令 pwd，获取当前工作目录路径。
# C.root_dir：获取当前工作目录的绝对路径。
# C.abs_dir：获取当前脚本所在的绝对路径。
remoteip = os.popen('pwd').read()
C.root_dir = os.path.abspath(os.path.join(os.getcwd(), './'))
C.abs_dir = osp.realpath(".")

# Dataset config  设置数据集相关配置
"""Dataset Path"""
C.dataset_name = 'WHU'
C.root_folder = '/mnt/newdisk/zyy/data/cd-dataset/WHU-CD-256'
C.A_format = '.png'
C.B_format = '.png'
C.gt_format = '.png'
C.is_test = False
C.num_train_imgs = 5947
C.num_eval_imgs = 743
C.num_classes = 2
C.class_names =  ['background', 'change']

"""Image Config  设置图像相关配置"""
C.background = 255
C.image_height = 256
C.image_width = 256
C.norm_mean = np.array([0.485, 0.456, 0.406])
C.norm_std = np.array([0.229, 0.224, 0.225])

""" Settings for network, this would be different for each kind of model  设置网络配置"""
C.backbone = 'sigma_small' # sigma_tiny / sigma_small / sigma_base
C.pretrained_model = None # do not need to change  预训练模型路径（None 表示不使用预训练模型）
# C.pretrained_model = '/mnt/newdisk/zyy/code/M-CD/pretrained/vssm_small_0229_ckpt_epoch_222.pth'
C.decoder = 'MambaDecoder' # 'MLPDecoder'
C.decoder_embed_dim = 512  # 解码器的嵌入维度
C.optimizer = 'AdamW'

"""Train Config  设置训练相关配置"""
C.lr = 6e-5
C.lr_power = 0.9
C.momentum = 0.9
C.weight_decay = 0.01
C.batch_size = 8
C.nepochs = 150
C.niters_per_epoch = C.num_train_imgs // C.batch_size  + 1
C.num_workers = 16
C.train_scale_array = [1]
C.train_scale_array = None
C.warm_up_epoch = 10

# 设置 BatchNorm 相关配置
C.fix_bias = True
C.bn_eps = 1e-3
C.bn_momentum = 0.1

"""Eval Config  设置评估相关配置"""
# C.eval_iter = 1
C.eval_stride_rate = 2 / 3
C.eval_scale_array = [1] 
C.eval_flip = False
C.eval_crop_size = [256, 256]

"""Store Config  设置检查点相关配置"""
C.checkpoint_start_epoch = 5
C.checkpoint_step = 5

"""Path Config"""
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
add_path(osp.join(C.root_dir))

# 设置日志和检查点目录
# C.log_dir：日志目录路径。
# C.tb_dir：TensorBoard 日志目录。
# C.checkpoint_dir：检查点目录。
# exp_time：当前时间，用于命名日志文件。
# C.log_file：当前运行日志文件路径。
# C.link_log_file：最新日志文件的符号链接路径。
# C.val_log_file：评估日志文件路径。
# C.link_val_log_file：最新评估日志文件的符号链接路径。
C.log_dir = osp.abspath('log_final/log_WHU/' + 'log_' + C.dataset_name + '_' + C.backbone + '_' + 'conmb_cvssdecoder')
C.tb_dir = osp.abspath(osp.join(C.log_dir, "tb"))
C.log_dir_link = C.log_dir
C.checkpoint_dir = osp.abspath(osp.join(C.log_dir, "checkpoint"))

exp_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
C.log_file = C.log_dir + '/log_' + exp_time + '.log'
C.link_log_file = C.log_file + '/log_last.log'
C.val_log_file = C.log_dir + '/val_' + exp_time + '.log'
C.link_val_log_file = C.log_dir + '/val_last.log'

if __name__ == '__main__':
    print(config.nepochs)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-tb', '--tensorboard', default=False, action='store_true')
    args = parser.parse_args()

    # if args.tensorboard:
        # open_tensorboard()