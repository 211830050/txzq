#
# import os
# import argparse
# import numpy as np
# from tqdm import tqdm
#
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.optim import lr_scheduler
# import torchvision
# from torchvision import transforms
# from torch.utils.data import DataLoader, Subset
#
# from skimage.metrics import peak_signal_noise_ratio as psnr
#
# from model import UResNet_P, Edge_Detector
# from dataloader import TrainDataSet
#
# torch.backends.cudnn.benchmark = True
#
# def train(config):
#     # 1. 设备
#     device = torch.device(f"cuda:{config.cuda_id}"
#                           if torch.cuda.is_available() else "cpu")
#     print("Using device:", device)
#
#     # 2. 模型与优化器
#     model = UResNet_P().to(device)
#     # 冻结 edge detector，仅用于 loss 计算
#     edge_detector = Edge_Detector().to(device)
#     for p in edge_detector.parameters():
#         p.requires_grad = False
#
#     if config.loss_type == 'MSE':
#         criterion = nn.MSELoss().to(device)
#     else:  # 'L1'
#         criterion = nn.L1Loss().to(device)
#
#     optimizer = optim.Adam(model.parameters(), lr=config.lr)
#     scheduler = lr_scheduler.StepLR(
#         optimizer,
#         step_size=config.step_size,
#         gamma=config.decay_rate
#     )
#
#     # 3. 数据预处理
#     tsfms = transforms.Compose([
#         transforms.Resize((config.resize, config.resize)),
#         transforms.ToTensor()
#     ])
#
#     # 4. 训练集 & 验证集
#     train_dataset = TrainDataSet(
#         config.input_images_path,
#         config.label_images_path,
#         tsfms
#     )
#     val_dataset = TrainDataSet(
#         config.val_input_path,
#         config.val_label_path,
#         tsfms
#     )
#
#     # 可选快速 smoke 测试
#     if config.smoke_test:
#         subset = min(config.smoke_size, len(train_dataset))
#         train_dataset = Subset(train_dataset, list(range(subset)))
#         print(f"Smoke test mode: using first {subset} samples")
#
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=config.batch_size,
#         shuffle=True,
#         num_workers=config.num_workers,
#         pin_memory=True
#     )
#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=config.batch_size,
#         shuffle=False,
#         num_workers=config.num_workers,
#         pin_memory=True
#     )
#
#     # 5. 训练 + 验证循环
#     for epoch in range(1, config.num_epochs + 1):
#         model.train()
#         loop = tqdm(train_loader, desc=f"Epoch {epoch}/{config.num_epochs}")
#         for inp, gt in loop:
#             inp, gt = inp.to(device), gt.to(device)
#
#             optimizer.zero_grad()
#             pred, edge_map = model(inp)
#             # 网络内部已 sigmoid，直接 clamp
#             pred = torch.clamp(pred, 0, 1)
#
#             img_loss  = criterion(pred, gt)
#             edge_tgt  = edge_detector(gt)
#             edge_loss = criterion(edge_map, edge_tgt)
#
#             if config.train_mode in ['P-S', 'P-A']:
#                 loss = img_loss + config.edge_weight * edge_loss
#             else:
#                 loss = img_loss
#
#             loss.backward()
#             optimizer.step()
#             loop.set_postfix(loss=f"{loss.item():.4f}")
#
#         scheduler.step()
#
#         # —— 验证 ——
#         model.eval()
#         psnr_vals = []
#         with torch.no_grad():
#             for inp, gt in val_loader:
#                 inp, gt = inp.to(device), gt.to(device)
#                 pred, _ = model(inp)
#                 pred = torch.clamp(pred, 0, 1)
#
#                 # 逐张计算 PSNR
#                 for b in range(pred.size(0)):
#                     pr = pred[b].cpu().permute(1,2,0).numpy()
#                     gr = gt[b].cpu().permute(1,2,0).numpy()
#                     psnr_vals.append(psnr(gr, pr, data_range=1))
#
#         avg_psnr = np.mean(psnr_vals)
#         print(f"[Epoch {epoch}] Validation PSNR: {avg_psnr:.2f} dB")
#
#         # —— 保存模型 ——
#         if epoch % config.snapshot_freq == 0:
#             os.makedirs(config.snapshots_folder, exist_ok=True)
#             ckpt_path = os.path.join(
#                 config.snapshots_folder,
#                 f"model_epoch_{epoch}.ckpt"
#             )
#             torch.save(model.state_dict(), ckpt_path)
#             print(f"Saved checkpoint: {ckpt_path}")
#
#     print("Training complete.")
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#
#     # 数据路径
#     parser.add_argument('--input_images_path', type=str,
#                         default='./dataset/train/input/',
#                         help='训练集输入图像路径')
#     parser.add_argument('--label_images_path', type=str,
#                         default='./dataset/train/label/',
#                         help='训练集标签图像路径')
#     parser.add_argument('--val_input_path', type=str,
#                         default='./dataset/val/input/',
#                         help='验证集输入图像路径')
#     parser.add_argument('--val_label_path', type=str,
#                         default='./dataset/val/label/',
#                         help='验证集标签图像路径')
#
#     # 超参数
#     parser.add_argument('--lr', type=float, default=2e-4,
#                         help='初始学习率')
#     parser.add_argument('--decay_rate', type=float, default=0.8,
#                         help='学习率衰减率')
#     parser.add_argument('--step_size', type=int, default=31,
#                         help='学习率衰减步长（epoch）')
#     parser.add_argument('--loss_type', type=str, default='MSE',
#                         choices=['MSE','L1'],
#                         help='主损失类型')
#     parser.add_argument('--num_epochs', type=int, default=620,
#                         help='训练总 epoch 数')
#     parser.add_argument('--train_mode', type=str,
#                         default='P-A',
#                         choices=['N','P-S','P-A'],
#                         help="训练模式：'N','P-S' 或 'P-A'")
#     parser.add_argument('--edge_weight', type=float, default=1.2,
#                         help='边缘损失权重')
#
#     # 其它配置
#     parser.add_argument('--batch_size', type=int, default=8)
#     parser.add_argument('--resize', type=int, default=256,
#                         help='Resize 大小')
#     parser.add_argument('--cuda_id', type=int, default=0)
#     parser.add_argument('--snapshot_freq', type=int, default=31,
#                         help='多少 epoch 保存一次模型')
#     parser.add_argument('--snapshots_folder', type=str, default='./snapshots/',
#                         help='模型保存目录')
#     parser.add_argument('--num_workers', type=int, default=16)
#     parser.add_argument('--smoke_test', action='store_true',
#                         help='快速 smoke 测试模式')
#     parser.add_argument('--smoke_size', type=int, default=500,
#                         help='smoke 测试子集大小')
#
#     config = parser.parse_args()
#     train(config)


# import os
# import argparse
# import numpy as np
# from tqdm import tqdm
#
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.optim import lr_scheduler
# import torchvision
# from torchvision import transforms
# from torch.utils.data import DataLoader, Subset
# from torch.nn import functional as F
#
# from model import UResNet_P, Edge_Detector
# from dataloader import TrainDataSet
#
# # 保持 cudnn 高效
# torch.backends.cudnn.benchmark = True
#
# def train(config):
#     device = torch.device(f"cuda:{config.cuda_id}" if torch.cuda.is_available() else "cpu")
#     print("Using device:", device)
#
#     # 模型和边缘检测器
#     model = UResNet_P().to(device)
#     edge_detector = Edge_Detector().to(device)
#     for p in edge_detector.parameters():
#         p.requires_grad = False
#
#     # 损失与优化器
#     criterion = nn.MSELoss().to(device) if config.loss_type == 'MSE' else nn.L1Loss().to(device)
#     optimizer = optim.Adam(model.parameters(), lr=config.lr)
#     scheduler = lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.decay_rate)
#
#     # 数据增广 + 转换
#     tsfms = transforms.Compose([
#         transforms.Resize((config.resize, config.resize)),
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomVerticalFlip(),
#         transforms.RandomRotation(15),
#         transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
#         transforms.ToTensor()
#     ])
#
#     # 训练/验证集加载
#     train_ds = TrainDataSet(config.input_images_path, config.label_images_path, tsfms)
#     val_ds   = TrainDataSet(config.val_input_path,     config.val_label_path,     tsfms)
#
#     if config.smoke_test:
#         subset = min(config.smoke_size, len(train_ds))
#         train_ds = Subset(train_ds, list(range(subset)))
#         print(f"Smoke test: using {subset} samples")
#
#     train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True,
#                               num_workers=config.num_workers, pin_memory=True)
#     val_loader   = DataLoader(val_ds,   batch_size=config.batch_size, shuffle=False,
#                               num_workers=config.num_workers, pin_memory=True)
#
#     # 迭代
#     for epoch in range(1, config.num_epochs+1):
#         model.train()
#         pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.num_epochs}")
#         for inp, gt in pbar:
#             inp, gt = inp.to(device), gt.to(device)
#
#             optimizer.zero_grad()
#             pred, edge_map = model(inp)
#             # 全尺度图像损失
#             img_loss = criterion(pred, gt)
#
#             # 多尺度边缘损失
#             # 全分辨率
#             edge_full = criterion(edge_map, edge_detector(gt))
#             # 半分辨率
#             gt_h = F.interpolate(gt, scale_factor=0.5, mode='bilinear', align_corners=False)
#             em_h = F.interpolate(edge_map, size=gt_h.shape[2:], mode='bilinear', align_corners=False)
#             edge_h = criterion(em_h, edge_detector(gt_h))
#             # 四分之一分辨率
#             gt_q = F.interpolate(gt, scale_factor=0.25, mode='bilinear', align_corners=False)
#             em_q = F.interpolate(edge_map, size=gt_q.shape[2:], mode='bilinear', align_corners=False)
#             edge_q = criterion(em_q, edge_detector(gt_q))
#
#             edge_loss = edge_full + 0.5 * edge_h + 0.25 * edge_q
#
#             # 总损失
#             if config.train_mode in ['P-S','P-A']:
#                 loss = img_loss + config.edge_weight * edge_loss
#             else:
#                 loss = img_loss
#
#             loss.backward()
#             optimizer.step()
#             pbar.set_postfix(loss=f"{loss.item():.4f}")
#
#         scheduler.step()
#
#         # 验证
#         model.eval()
#         psnr_vals = []
#         with torch.no_grad():
#             for inp, gt in val_loader:
#                 inp, gt = inp.to(device), gt.to(device)
#                 pred, _ = model(inp)
#                 for b in range(pred.size(0)):
#                     pr = pred[b].cpu().permute(1,2,0).numpy()
#                     gr = gt[b].cpu().permute(1,2,0).numpy()
#                     psnr_vals.append(10 * np.log10(1.0 / ((pr-gr)**2).mean()))
#         print(f"[Epoch {epoch}] Val PSNR: {np.mean(psnr_vals):.2f} dB")
#
#         # 保存
#         if epoch % config.snapshot_freq == 0:
#             os.makedirs(config.snapshots_folder, exist_ok=True)
#             path = os.path.join(config.snapshots_folder, f"model_epoch_{epoch}.ckpt")
#             torch.save(model.state_dict(), path)
#             print("Saved:", path)
#
#     print("Training complete.")
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     # 路径
#     parser.add_argument('--input_images_path', type=str, default='./dataset/train/input/')
#     parser.add_argument('--label_images_path', type=str, default='./dataset/train/label/')
#     parser.add_argument('--val_input_path',   type=str, default='./dataset/val/input/')
#     parser.add_argument('--val_label_path',   type=str, default='./dataset/val/label/')
#     # 超参
#     parser.add_argument('--lr',           type=float, default=2e-4)
#     parser.add_argument('--decay_rate',   type=float, default=0.8)
#     parser.add_argument('--step_size',    type=int,   default=50)
#     parser.add_argument('--num_epochs',   type=int,   default=500)
#     parser.add_argument('--loss_type',    type=str,   default='MSE', choices=['MSE','L1'])
#     parser.add_argument('--train_mode',   type=str,   default='P-A', choices=['N','P-S','P-A'])
#     parser.add_argument('--edge_weight',  type=float, default=1.2)
#     # 其它
#     parser.add_argument('--batch_size',      type=int, default=8)
#     parser.add_argument('--resize',          type=int, default=256)
#     parser.add_argument('--cuda_id',         type=int, default=0)
#     parser.add_argument('--snapshot_freq',   type=int, default=50)
#     parser.add_argument('--snapshots_folder',type=str, default='./snapshots/')
#     parser.add_argument('--num_workers',     type=int, default=16)
#     parser.add_argument('--smoke_test',      action='store_true')
#     parser.add_argument('--smoke_size',      type=int, default=500)
#
#     config = parser.parse_args()
#     train(config)



# -*- coding: utf-8 -*-
import os
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from torch.nn import functional as F

# SSIM Loss
from pytorch_msssim import ssim

from model import UResNet_P, Edge_Detector
from dataloader import TrainDataSet

torch.backends.cudnn.benchmark = True

def train(config):
    # 1. Device
    device = torch.device(f"cuda:{config.cuda_id}" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 2. Model
    model = UResNet_P().to(device)
    edge_detector = Edge_Detector().to(device)
    for p in edge_detector.parameters(): p.requires_grad = False

    # 3. Losses & optimizer
    criterion = nn.MSELoss().to(device) if config.loss_type == 'MSE' else nn.L1Loss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.num_epochs, eta_min=config.lr*0.01)

    # 4. Transforms
    tsfms_train = transforms.Compose([
        transforms.Resize((config.resize, config.resize)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor()
    ])
    tsfms_val = transforms.Compose([
        transforms.Resize((config.resize, config.resize)),
        transforms.ToTensor()
    ])

    # 5. Datasets
    train_ds = TrainDataSet(config.input_images_path, config.label_images_path, tsfms_train)
    val_ds   = TrainDataSet(config.val_input_path,     config.val_label_path,     tsfms_val)
    if config.smoke_test:
        subset = min(config.smoke_size, len(train_ds))
        train_ds = Subset(train_ds, list(range(subset)))
        print(f"Smoke test: using {subset} samples")

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True,
                              num_workers=config.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=config.batch_size, shuffle=False,
                              num_workers=config.num_workers, pin_memory=True)

    # 6. Training loop
    for epoch in range(1, config.num_epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.num_epochs}")
        for inp, gt in pbar:
            inp, gt = inp.to(device), gt.to(device)

            optimizer.zero_grad()
            pred, edge_map = model(inp)

            # Reconstruction loss
            img_loss = criterion(pred, gt)
            # Edge loss
            edge_full = criterion(edge_map, edge_detector(gt))
            # SSIM loss
            ssim_loss = 1 - ssim(pred, gt, data_range=1.0, size_average=True)

            # total loss = img + edge + ssim
            loss = img_loss \
                   + config.edge_weight * edge_full \
                   + config.ssim_weight * ssim_loss

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'img': f"{img_loss.item():.4f}",
                'edge': f"{edge_full.item():.4f}",
                'ssim': f"{ssim_loss.item():.4f}"
            })

        scheduler.step()

        # Validation PSNR/SSIM
        model.eval()
        psnr_vals, ssim_vals = [], []
        with torch.no_grad():
            for inp, gt in val_loader:
                inp, gt = inp.to(device), gt.to(device)
                pred, _ = model(inp)
                pred = torch.clamp(pred, 0, 1)
                # batch
                for b in range(pred.size(0)):
                    pr = pred[b].cpu().permute(1,2,0).numpy()
                    gr = gt[b].cpu().permute(1,2,0).numpy()
                    mse = ((pr-gr)**2).mean()
                    psnr_vals.append(10 * np.log10(1.0/mse) if mse>0 else float('inf'))
                # SSIM per batch
                ssim_vals.append(ssim(pred, gt, data_range=1.0, size_average=True).item())
        print(f"[Epoch {epoch}] Val PSNR: {np.mean(psnr_vals):.2f} dB, SSIM: {np.mean(ssim_vals):.4f}")

        # 7. Checkpoint
        if epoch % config.snapshot_freq == 0:
            os.makedirs(config.snapshots_folder, exist_ok=True)
            path = os.path.join(config.snapshots_folder, f"epoch{epoch}_psnr{np.mean(psnr_vals):.2f}.ckpt")
            torch.save(model.state_dict(), path)
            print("Saved:", path)

    print("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # data paths
    parser.add_argument('--input_images_path', type=str, default='./dataset/train/input/')
    parser.add_argument('--label_images_path', type=str, default='./dataset/train/label/')
    parser.add_argument('--val_input_path',   type=str, default='./dataset/val/input/')
    parser.add_argument('--val_label_path',   type=str, default='./dataset/val/label/')
    # hyperparams
    parser.add_argument('--lr',           type=float, default=2e-4)
    parser.add_argument('--num_epochs',   type=int,   default=500)
    parser.add_argument('--loss_type',    type=str,   default='MSE', choices=['MSE','L1'])
    parser.add_argument('--train_mode',   type=str,   default='P-A', choices=['P-S','P-A'])
    parser.add_argument('--edge_weight',  type=float, default=2.0)
    parser.add_argument('--ssim_weight',  type=float, default=0.1, help='weight for SSIM loss')
    parser.add_argument('--grad_clip',    type=float, default=5.0)
    # other
    parser.add_argument('--batch_size',      type=int, default=8)
    parser.add_argument('--resize',          type=int, default=256)
    parser.add_argument('--cuda_id',         type=int, default=0)
    parser.add_argument('--snapshot_freq',   type=int, default=50)
    parser.add_argument('--snapshots_folder',type=str, default='./snapshots/')
    parser.add_argument('--num_workers',     type=int, default=16)
    parser.add_argument('--smoke_test',      action='store_true')
    parser.add_argument('--smoke_size',      type=int, default=500)

    config = parser.parse_args()
    train(config)
