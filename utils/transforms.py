import warnings
import torch.nn as nn
import torch.nn.functional as F

import cv2
import numpy as np
import numbers
import random
import collections

try:
    collections = collections.abc
except AttributeError:
    pass

# 定义了一系列用于图像增强和预处理的函数，常用于深度学习中的图像处理。

# 确保 shape 是一个二维元组。如果是单个数字，转为 (shape, shape)；如果是元组，则提取高度和宽度。
def get_2dshape(shape, *, zero=True):
    if not isinstance(shape, collections.Iterable):
        shape = int(shape)
        shape = (shape, shape)
    else:
        h, w = map(int, shape)
        shape = (h, w)
    if zero:
        minv = 0
    else:
        minv = 1

    assert min(shape) >= minv, 'invalid shape: {}'.format(shape)
    return shape


# 随机裁剪图像，并根据指定大小进行填充。保证裁剪的位置和大小合法。最后调用 pad_image_to_shape 函数将图像填充到所需大小。
def random_crop_pad_to_shape(img, crop_pos, crop_size, pad_label_value):
    h, w = img.shape[:2]
    start_crop_h, start_crop_w = crop_pos
    assert ((start_crop_h < h) and (start_crop_h >= 0))
    assert ((start_crop_w < w) and (start_crop_w >= 0))

    crop_size = get_2dshape(crop_size)
    crop_h, crop_w = crop_size

    img_crop = img[start_crop_h:start_crop_h + crop_h,
               start_crop_w:start_crop_w + crop_w, ...]

    img_, margin = pad_image_to_shape(img_crop, crop_size, cv2.BORDER_CONSTANT,
                                      pad_label_value)

    return img_, margin

# 根据输入的原始大小 ori_size 和裁剪大小 crop_size，随机生成裁剪的起始位置。
def generate_random_crop_pos(ori_size, crop_size):
    ori_size = get_2dshape(ori_size)
    h, w = ori_size

    crop_size = get_2dshape(crop_size)
    crop_h, crop_w = crop_size

    pos_h, pos_w = 0, 0

    if h > crop_h:
        pos_h = random.randint(0, h - crop_h + 1)

    if w > crop_w:
        pos_w = random.randint(0, w - crop_w + 1)

    return pos_h, pos_w

# 将图像填充到指定大小。根据 shape 和图像当前大小计算出需要填充的像素数。调用 cv2.copyMakeBorder 执行填充操作。
def pad_image_to_shape(img, shape, border_mode, value):
    margin = np.zeros(4, np.uint32)
    shape = get_2dshape(shape)
    pad_height = shape[0] - img.shape[0] if shape[0] - img.shape[0] > 0 else 0
    pad_width = shape[1] - img.shape[1] if shape[1] - img.shape[1] > 0 else 0

    margin[0] = pad_height // 2
    margin[1] = pad_height // 2 + pad_height % 2
    margin[2] = pad_width // 2
    margin[3] = pad_width // 2 + pad_width % 2

    img = cv2.copyMakeBorder(img, margin[0], margin[1], margin[2], margin[3], border_mode, value=value)

    return img, margin

# 将图像的大小填充为 multiple 的倍数。通过对图像大小进行规范化，将其调整到最接近的 multiple 的倍数。
def pad_image_size_to_multiples_of(img, multiple, pad_value):
    h, w = img.shape[:2]
    d = multiple

    def canonicalize(s):
        v = s // d
        return (v + (v * d != s)) * d

    th, tw = map(canonicalize, (h, w))

    return pad_image_to_shape(img, (th, tw), cv2.BORDER_CONSTANT, pad_value)

# 根据图像的短边调整图像大小。
def resize_ensure_shortest_edge(img, edge_length,
                                interpolation_mode=cv2.INTER_LINEAR):
    assert isinstance(edge_length, int) and edge_length > 0, edge_length
    h, w = img.shape[:2]
    if h < w:
        ratio = float(edge_length) / h
        th, tw = edge_length, max(1, int(ratio * w))
    else:
        ratio = float(edge_length) / w
        th, tw = max(1, int(ratio * h)), edge_length
    img = cv2.resize(img, (tw, th), interpolation_mode)

    return img

# 随机缩放图像及其对应的标签（和额外模态数据）。
def random_scale(img, gt, scales):
    scale = random.choice(scales)
    sh = int(img.shape[0] * scale)
    sw = int(img.shape[1] * scale)
    img = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_LINEAR)
    gt = cv2.resize(gt, (sw, sh), interpolation=cv2.INTER_NEAREST)
    
    return img, gt, scale

def random_scale_rgbx(img, gt, modal_x, scales):
    scale = random.choice(scales)
    sh = int(img.shape[0] * scale)
    sw = int(img.shape[1] * scale)
    img = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_LINEAR)
    gt = cv2.resize(gt, (sw, sh), interpolation=cv2.INTER_NEAREST)
    modal_x = cv2.resize(modal_x, (sw, sh), interpolation=cv2.INTER_LINEAR)

    return img, gt, modal_x, scale

def random_scale_with_length(img, gt, length):
    size = random.choice(length)
    sh = size
    sw = size
    img = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_LINEAR)
    gt = cv2.resize(gt, (sw, sh), interpolation=cv2.INTER_NEAREST)

    return img, gt, size

# 随机水平翻转图像及其标签。
def random_mirror(img, gt):
    if random.random() >= 0.5:
        img = cv2.flip(img, 1)
        gt = cv2.flip(gt, 1)

    return img, gt,

# 对图像进行随机角度的旋转。
def random_rotation(img, gt):
    angle = random.random() * 20 - 10
    h, w = img.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    img = cv2.warpAffine(img, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR)
    gt = cv2.warpAffine(gt, rotation_matrix, (w, h), flags=cv2.INTER_NEAREST)

    return img, gt

# 对图像进行高斯模糊，随机选择模糊核的大小。
def random_gaussian_blur(img):
    gauss_size = random.choice([1, 3, 5, 7])
    if gauss_size > 1:
        # do the gaussian blur
        img = cv2.GaussianBlur(img, (gauss_size, gauss_size), 0)

    return img

# 居中裁剪图像到指定大小。
def center_crop(img, shape):
    h, w = shape[0], shape[1]
    y = (img.shape[0] - h) // 2
    x = (img.shape[1] - w) // 2
    return img[y:y + h, x:x + w]

# 随机裁剪图像和标签。
def random_crop(img, gt, size):
    if isinstance(size, numbers.Number):
        size = (int(size), int(size))
    else:
        size = size

    h, w = img.shape[:2]
    crop_h, crop_w = size[0], size[1]

    if h > crop_h:
        x = random.randint(0, h - crop_h + 1)
        img = img[x:x + crop_h, :, :]
        gt = gt[x:x + crop_h, :]

    if w > crop_w:
        x = random.randint(0, w - crop_w + 1)
        img = img[:, x:x + crop_w, :]
        gt = gt[:, x:x + crop_w]

    return img, gt

# 将图像进行归一化处理，标准化到指定的均值和标准差。
def normalize(img, mean, std):
    # pytorch pretrained model need the input range: 0-1
    img = img.astype(np.float64) / 255.0
    img = img - mean
    img = img / std
    return img
