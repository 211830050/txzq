# #!/usr/bin/env python
# # enhance_images_fixed_params.py
#
# import os
# from PIL import Image
# from tqdm import tqdm
#
# import torch
# from torchvision import transforms
# from torch.utils.data import Dataset, DataLoader
#
# from model import UNetPlus
#
# # -----------------------------
# # 固定参数配置
# # -----------------------------
# INPUT_FOLDER = './dataset/t/'
# OUTPUT_FOLDER = './results/'
# CKPT_PATH = './snapshots/best.pth'
# BATCH_SIZE = 8
# RESIZE = 256
# CUDA_ID = 0
# NUM_WORKERS = 4
# BASE_CHANNELS = 64
#
# # -----------------------------
# # 数据集定义
# # -----------------------------
# class ImageDataset(Dataset):
#     """无标签的图片文件夹 Dataset"""
#     def __init__(self, folder, transform):
#         self.folder = folder
#         self.transform = transform
#         # 只取常见的图像格式
#         self.fnames = sorted([
#             f for f in os.listdir(folder)
#             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
#         ])
#
#     def __len__(self):
#         return len(self.fnames)
#
#     def __getitem__(self, idx):
#         fname = self.fnames[idx]
#         path = os.path.join(self.folder, fname)
#         img = Image.open(path).convert('RGB')
#         return self.transform(img), fname
#
# # -----------------------------
# # 主流程
# # -----------------------------
# def enhance():
#     # 设备
#     device = torch.device(f"cuda:{CUDA_ID}" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")
#
#     # 模型加载
#     model = UNetPlus(base_channels=BASE_CHANNELS).to(device)
#     ckpt = torch.load(CKPT_PATH, map_location=device)
#     model.load_state_dict(ckpt)
#     model.eval()
#
#     # 预处理：Resize + ToTensor
#     transform = transforms.Compose([
#         transforms.Resize((RESIZE, RESIZE)),
#         transforms.ToTensor()
#     ])
#
#     # 数据加载
#     ds = ImageDataset(INPUT_FOLDER, transform)
#     loader = DataLoader(ds,
#                         batch_size=BATCH_SIZE,
#                         shuffle=False,
#                         num_workers=NUM_WORKERS,
#                         pin_memory=True)
#
#     # 创建输出目录
#     os.makedirs(OUTPUT_FOLDER, exist_ok=True)
#
#     # 推理并保存
#     with torch.no_grad():
#         for batch, fnames in tqdm(loader, desc="Enhancing Images"):
#             batch = batch.to(device)
#             out = model(batch)            # [B,3,H,W], 归一化到 [0,1]
#             out = torch.clamp(out, 0, 1)
#
#             for i, fname in enumerate(fnames):
#                 img_tensor = out[i].cpu().permute(1, 2, 0)  # HWC
#                 img_np = (img_tensor.numpy() * 255.0).round().astype('uint8')
#                 save_path = os.path.join(OUTPUT_FOLDER, fname)
#                 Image.fromarray(img_np).save(save_path)
#
#     print("All done. Enhanced images saved to", OUTPUT_FOLDER)
#
# if __name__ == "__main__":
#     enhance()


# -*- coding: utf-8 -*-
import os
from PIL import Image
from tqdm import tqdm

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from model import UNetPlus  # 你的模型定义

# -----------------------------
# 配置区
# -----------------------------
INPUT_FOLDER  = './dataset/t/'
OUTPUT_FOLDER = './results/'
CKPT_PATH     = './snapshots/model60.pth'
BATCH_SIZE    = 8
RESIZE        = 256
CUDA_ID       = 0
NUM_WORKERS   = 0     # 改为 0，避免 Windows 下多进程问题
PIN_MEMORY    = False # 关闭 pin_memory
BASE_CHANNELS = 64

# -----------------------------
# Dataset 定义
# -----------------------------
class ImageDataset(Dataset):
    """无标签图片文件夹加载，只支持常见后缀"""
    def __init__(self, folder, transform):
        self.folder = folder
        self.transform = transform
        self.fnames = sorted([
            f for f in os.listdir(folder)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
        ])

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img = Image.open(os.path.join(self.folder, fname)).convert('RGB')
        return self.transform(img), fname

def enhance():
    # 1. 设备
    device = torch.device(f"cuda:{CUDA_ID}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. 模型加载
    model = UNetPlus(base_channels=BASE_CHANNELS).to(device)
    ckpt = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()

    # 3. 数据预处理 + DataLoader
    transform = transforms.Compose([
        transforms.Resize((RESIZE, RESIZE)),
        transforms.ToTensor()
    ])
    ds = ImageDataset(INPUT_FOLDER, transform)
    loader = DataLoader(ds,
                        batch_size=BATCH_SIZE,
                        shuffle=False,
                        num_workers=NUM_WORKERS,
                        pin_memory=PIN_MEMORY)

    # 4. 创建输出目录
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # 5. 推理并保存
    with torch.no_grad():
        for batch, fnames in tqdm(loader, desc="Enhancing Images"):
            batch = batch.to(device)
            out = model(batch)            # UNetPlus 返回 [B,3,H,W]
            out = torch.clamp(out, 0, 1)

            for i, fname in enumerate(fnames):
                img_tensor = out[i].cpu().permute(1, 2, 0)  # HWC
                img_np = (img_tensor.numpy() * 255.0).round().astype('uint8')
                save_path = os.path.join(OUTPUT_FOLDER, fname)
                Image.fromarray(img_np).save(save_path)

    print("All done. Enhanced images saved to", OUTPUT_FOLDER)

if __name__ == "__main__":
    # Windows 下多进程支持
    import multiprocessing
    multiprocessing.freeze_support()

    enhance()
