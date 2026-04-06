import numpy as np
import cv2
import scipy.io as sio

# 这段代码包含了一些图像处理和评估的函数，主要用于语义分割任务的预测结果可视化，以及计算和打印模型的性能指标

# 根据预测结果 pred 为图像 img 着色 img: 需要着色的图像。  对于每个类别，如果该类别不是背景，就把 pred 中预测为该类别的像素设置为对应的颜色。
# pred: 模型的预测结果  show255: 如果为 True，则将 gt 中属于背景的像素设置为 255（白色）
def set_img_color(colors, background, img, pred, gt, show255=False):
    for i in range(0, len(colors)):
        if i != background:
            img[np.where(pred == i)] = colors[i]
    if show255:
        img[np.where(gt == background)] = 255
    return img

def show_prediction(colors, background, img, pred, gt):    # 使用 set_img_color 函数对图像进行着色，并返回处理后的图像
    im = np.array(img, np.uint8)
    set_img_color(colors, background, im, pred, gt)
    final = np.array(im)
    return final

# 将多个预测结果和真实标签在同一图像中进行展示  clean: 原始图像，但在此函数中未使用。  *pds: 可变参数，表示多个预测结果。
def show_img(colors, background, img, clean, gt, *pds):
    im1 = np.array(img, np.uint8)
    # set_img_color(colors, background, im1, clean, gt)
    final = np.array(im1)
    # the pivot black bar
    pivot = np.zeros((im1.shape[0], 15, 3), dtype=np.uint8)
    for pd in pds:
        im = np.array(img, np.uint8)
        # pd[np.where(gt == 255)] = 255
        set_img_color(colors, background, im, pd, gt)
        final = np.column_stack((final, pivot))
        final = np.column_stack((final, im))

    im = np.array(img, np.uint8)
    set_img_color(colors, background, im, gt, True)
    final = np.column_stack((final, pivot))
    final = np.column_stack((final, im))
    return final

# 随机生成 class_num 个颜色，每个颜色用一个 RGB 值表示。
def get_colors(class_num):
    colors = []
    for i in range(class_num):
        colors.append((np.random.random((1, 3)) * 255).tolist()[0])

        # # 保证颜色不接近黑色
        # if np.sum(color) < 50:  # 如果颜色太暗
        #     color = [c + 50 for c in color]  # 增加亮度
        # colors.append(color)

    return colors

# 从 color150.mat 文件中加载 ADE20K 数据集的颜色映射，并进行处理。
def get_ade_colors():
    colors = sio.loadmat('./color150.mat')['colors']
    colors = colors[:,::-1,]
    colors = np.array(colors).astype(int).tolist()
    colors.insert(0,[0,0,0])

    return colors


def print_iou(iou, recall, precision, freq_IoU, mean_pixel_acc, pixel_acc, class_names=None, show_no_back=False, no_print=False):
    n = iou.size
    lines = []
    for i in range(n):
        if class_names is None:
            cls = 'Class %d:' % (i+1)
        else:
            cls = '%d %s' % (i+1, class_names[i])
        lines.append('%-8s\t%.3f%%' % (cls, iou[i] * 100))
    mean_IoU = np.nanmean(iou)
    mean_IoU_no_back = np.nanmean(iou[1:])
    if show_no_back:
        lines.append('----------     %-8s\t%.3f%%\t%-8s\t%.3f%%\t%-8s\t%.3f%%\t%-8s\t%.3f%%\t%-8s\t%.3f%%\t%-8s\t%.3f%%\t%-8s\t%.3f%%' % ('mean_IoU', mean_IoU * 100, 'precision', precision * 100, 'recall', recall * 100, 'mean_IU_no_back', mean_IoU_no_back * 100,
                                                                                                                'freq_IoU', freq_IoU * 100, 'mean_pixel_acc', mean_pixel_acc * 100, 'pixel_acc',pixel_acc * 100))
    else:
        lines.append('----------     %-8s\t%.3f%%\t%-8s\t%.3f%%\t%-8s\t%.3f%%\t%-8s\t%.3f%%\t%-8s\t%.3f%%\t%-8s\t%.3f%%' % ('mean_IoU', mean_IoU * 100, 'precision', precision * 100, 'recall', recall * 100, 'freq_IoU', freq_IoU * 100,
                                                                                                    'mean_pixel_acc', mean_pixel_acc * 100, 'pixel_acc',pixel_acc * 100))
    line = "\n".join(lines)
    if not no_print:
        print(line)
    return line


