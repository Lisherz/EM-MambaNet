import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from torchvision import transforms

class ContrastiveChangeDetectionDataset(Dataset):
    def __init__(self, t1_dir, t2_dir, label_dir, transform=None):
        self.t1_dir = t1_dir
        self.t2_dir = t2_dir
        self.label_dir = label_dir
        self.transform = transform

        self.filenames = os.listdir(label_dir)
        self.pos_pairs = []  # 正样本对: label == 0
        self.neg_pairs = []  # 负样本对: label == 1

        # 分类正负样本
        for name in self.filenames:
            label_path = os.path.join(label_dir, name)
            label = Image.open(label_path).convert('L')
            label_np = np.array(label)
            if np.mean(label_np) < 0.1:  # 大部分是未变化区域，归为正样本
                self.pos_pairs.append(name)
            elif np.mean(label_np) > 0.5:  # 大部分是变化区域，归为负样本
                self.neg_pairs.append(name)

        print(f"[INFO] 正样本数量: {len(self.pos_pairs)}, 负样本数量: {len(self.neg_pairs)}")

    def __len__(self):
        return min(len(self.pos_pairs), len(self.neg_pairs)) * 2

    def __getitem__(self, idx):
        # 一半采正样本对，一半采负样本对
        if idx % 2 == 0:
            # 正样本对: T1, T2 都是未变化区域
            name1, name2 = random.sample(self.pos_pairs, 2)
            label = 1  # 对比学习中，1 表示“相似”
        else:
            # 负样本对: 一个变化，一个未变化
            name1 = random.choice(self.pos_pairs)
            name2 = random.choice(self.neg_pairs)
            label = 0  # 对比学习中，0 表示“不相似”

        # 加载图像
        t1_1 = Image.open(os.path.join(self.t1_dir, name1)).convert('RGB')
        t2_1 = Image.open(os.path.join(self.t2_dir, name1)).convert('RGB')

        t1_2 = Image.open(os.path.join(self.t1_dir, name2)).convert('RGB')
        t2_2 = Image.open(os.path.join(self.t2_dir, name2)).convert('RGB')

        if self.transform:
            t1_1 = self.transform(t1_1)
            t2_1 = self.transform(t2_1)
            t1_2 = self.transform(t1_2)
            t2_2 = self.transform(t2_2)

        # 模态融合：你可以自行决定如何组合，例如 cat 或用多分支编码器
        # 这里只返回 (input1, input2, label)
        input1 = torch.cat([t1_1, t2_1], dim=0)  # 6-channel
        input2 = torch.cat([t1_2, t2_2], dim=0)

        return input1, input2, torch.tensor(label, dtype=torch.float32)

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

dataset = ContrastiveChangeDetectionDataset(
    t1_dir='/mnt/newdisk/zyy/data/multi/ShuGuang/A',
    t2_dir='/mnt/newdisk/zyy/data/multi/ShuGuang/B',
    label_dir='/mnt/newdisk/zyy/data/multi/ShuGuang/gt',
    transform=transform
)

dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

for batch in dataloader:
    input1, input2, labels = batch
    print(input1.shape, input2.shape, labels)
    break
