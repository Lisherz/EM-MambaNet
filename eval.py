import os
import cv2
import argparse
import numpy as np

import torch
import torch.nn as nn

from utils.pyt_utils import ensure_dir, link_file, load_model, parse_devices
from utils.visualize import print_iou, show_img
from engine.evaluator import Evaluator
from engine.logger import get_logger
from utils.metric import hist_info, compute_score
from dataloader.changeDataset import ChangeDataset
from models.builder import EncoderDecoder as segmodel
from dataloader.dataloader import ValPre
from PIL import Image

logger = get_logger()

# 单GPU评估
# CUDA_VISIBLE_DEVICES="0" python eval.py -d="0" -n "dataset_name" -e="epoch_number" -p="visualize_savedir"
# 多GPU评估
# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python eval.py -d="0,1,2,3,4,5,6,7" -n "dataset_name" -e="epoch_number" -p="visualize_savedir"


class SegEvaluator(Evaluator):
    # 添加
    def __init__(self, *args, **kwargs):
        super(SegEvaluator, self).__init__(*args, **kwargs)
        self.predictions = []  # 新增属性，用来存储预测结果

    def func_per_iteration(self, data, device, config):
        As = data['A']
        Bs = data['B']
        label = data['gt']
        name = data['fn']

        # Debugging: check if config is None or missing attributes
        assert config is not None, "Config object is None."
        assert hasattr(config, 'eval_crop_size'), "Config object is missing 'eval_crop_size'."
        assert hasattr(config, 'eval_stride_rate'), "Config object is missing 'eval_stride_rate'."

        # 调用 self.sliding_eval_rgbX() 方法生成预测结果 pred，并计算与标签的比较信息。
        pred = self.sliding_eval_rgbX(As, Bs, config.eval_crop_size, config.eval_stride_rate, device)
        print(pred.shape)

        # 确保 pred 是 NumPy 数组，而不是列表
        if isinstance(pred, list):
            pred = np.array(pred)  # 如果 pred 是列表，转换为 NumPy 数组
        if isinstance(pred, np.ndarray) and pred.ndim == 3:
            pred = pred.squeeze()  # 如果维度是 3D（例如：N x H x W），去掉额外的维度

        # 存储预测结果
        self.predictions.append(pred)

        hist_tmp, labeled_tmp, correct_tmp = hist_info(config.num_classes, pred, label)
        results_dict = {'hist': hist_tmp, 'labeled': labeled_tmp, 'correct': correct_tmp}

        if self.save_path is not None:
            ensure_dir(self.save_path)
            ensure_dir(self.save_path+'_color')

            fn = name + '.png'

            # 如果需要保存彩色结果
            # result_img = Image.fromarray(pred.astype(np.uint8) * 255, mode='P')
            # result_img = Image.fromarray(pred.astype(np.uint8) * 255, mode='L')
            # result_img.save(os.path.join(self.save_path+'_color', fn))

            # 保存二分类结果图像
            pred_image = (pred * 255).astype(np.uint8)  # Scale to 0-255 for saving
            cv2.imwrite(os.path.join(self.save_path, fn), pred_image)
            logger.info('Save the image ' + fn)

        if self.show_image:
            colors = self.dataset.get_class_colors()
            image = As
            clean = np.zeros(label.shape)
            comp_img = show_img(colors, config.background, image, clean,
                                label,
                                pred)
            cv2.imshow('comp_image', comp_img)
            cv2.waitKey(0)

        return results_dict

    def compute_metric(self, results):
        hist = np.zeros((self.config.num_classes, self.config.num_classes))
        correct = 0
        labeled = 0
        count = 0
        for d in results:
            hist += d['hist']
            correct += d['correct']
            labeled += d['labeled']
            count += 1

        iou, recall, precision, mean_IoU, _, freq_IoU, mean_pixel_acc, pixel_acc = compute_score(hist, correct, labeled)
        result_line = print_iou(iou, recall, precision, freq_IoU, mean_pixel_acc, pixel_acc,
                                self.dataset.class_names, show_no_back=False)
        return result_line, mean_IoU

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', default='last', type=str)
    parser.add_argument('-d', '--devices', default='0', type=str)
    parser.add_argument('-v', '--verbose', default=False, action='store_true')
    parser.add_argument('--show_image', '-s', default=False,
                        action='store_true')
    parser.add_argument('--save_path', '-p', default=None)
    parser.add_argument('--dataset_name', '-n', default='mfnet', type=str)
    parser.add_argument('--split', '-c', default='test', type=str)

    args = parser.parse_args()
    all_dev = parse_devices(args.devices)
    
    dataset_name = args.dataset_name
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

    network = segmodel(cfg=config, criterion=None, norm_layer=nn.BatchNorm2d)
    flops = network.flops()
    print("Gflops of the network: ", flops/(10**9))
    print("number of paramters: ", sum(p.numel() if p.requires_grad==True else 0 for p in network.parameters()))

    # data_setting = {'root': config.root_folder,
    #                 'A_format': config.A_format,
    #                 'B_format': config.B_format,
    #                 'gt_format': config.gt_format,
    #                 'class_names': config.class_names}

    # 修改的
    data_setting = {'root': config.root_folder,
                   'A_format': config.A_eval_format,
                   'B_format': config.B_eval_format,
                   'gt_format': config.gt_eval_format,
                   'class_names': config.class_names}
    val_pre = ValPre()
    dataset = ChangeDataset(data_setting, args.split, val_pre)
 
    with torch.no_grad():
        segmentor = SegEvaluator(dataset, config.num_classes, config.norm_mean,
                                 config.norm_std, network,
                                 config.eval_scale_array, config.eval_flip,
                                 all_dev, args.verbose, args.save_path,
                                 args.show_image, config)
        _, mean_IoU = segmentor.run_eval(config.checkpoint_dir, args.epochs, config.val_log_file,
                      config.link_val_log_file)