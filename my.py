import numpy as np
from PIL import Image

def calculate_region_ratio(label_path):
    """
    计算标签图像中变化区域（白色）和非变化区域（黑色）的比例。

    参数：
        label_path (str): 标签图像的路径，图像为黑白图像（0 表示黑，255 表示白）。

    返回：
        dict: 一个字典，包含变化区域和非变化区域的像素数及比例。
    """
    # 读取标签图像
    label_image = Image.open(label_path).convert('L')
    label_array = np.array(label_image)

    # 计算变化区域（白色，255）和非变化区域（黑色，0）的像素数
    white_pixel_count = np.sum(label_array == 255)
    black_pixel_count = np.sum(label_array == 0)

    # 计算总像素数
    total_pixel_count = white_pixel_count + black_pixel_count

    # 计算比例
    white_ratio = white_pixel_count / total_pixel_count
    black_ratio = black_pixel_count / total_pixel_count

    return {
        "变化区域像素数": white_pixel_count,
        "非变化区域像素数": black_pixel_count,
        "变化区域比例": white_ratio,
        "非变化区域比例": black_ratio
    }

# 示例：
label_path = "/mnt/newdisk/zyy/data/Gloucester/Gloucester_gt.png"  # 替换为你的标签图像路径
result = calculate_region_ratio(label_path)
print(result)

# California数据集：{'变化区域像素数': np.int64(19135), '非变化区域像素数': np.int64(418365),
# '变化区域比例': np.float64(0.043737142857142856), '非变化区域比例': np.float64(0.9562628571428572)}

# Gloucester数据集：