
# import torch
# import torchvision
# from torchvision import transforms
# import os
# import argparse
#
# from model import Edge_Detector, Res_Block, UResNet_P
# from dataloader import TestDataSet
#
# def test(config):
#     # 1. 设备配置
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     # 2. 实例化网络并加载 state_dict
#     test_model = UResNet_P().to(device)
#     ckpt = torch.load(config.snapshot_pth, map_location=device)
#     # 如果是 Lightning `.ckpt`，权重在 'state_dict' 键下
#     if 'state_dict' in ckpt:
#         ckpt = ckpt['state_dict']
#     test_model.load_state_dict(ckpt)
#     test_model.eval()
#
#     # 3. 创建输出目录
#     os.makedirs(config.output_pth, exist_ok=True)
#
#     # 4. 数据预处理与加载
#     transform = transforms.Compose([
#         transforms.ToTensor()
#     ])
#     testset = TestDataSet(config.test_pth, transform)
#     print(f"共加载测试图片数量：{len(testset)}")
#
#     loader = torch.utils.data.DataLoader(
#         testset, batch_size=config.batch_size, shuffle=False
#     )
#
#     # 5. 推理与保存
#     for i, (img, name) in enumerate(loader, 1):
#         with torch.no_grad():
#             img = img.to(device)
#
#             # 网络输出 residual（增量）
#             residual, _ = test_model(img)
#
#             # 把残差加回原图，并裁剪到 [0,1]
#             enhanced = img + residual
#             out = torch.clamp(enhanced, 0, 1)
#
#             # 打印范围调试
#             print(f"[{i}/{len(testset)}] 输入范围:  min={img.min().item():.4f}, max={img.max().item():.4f}")
#             print(f"[{i}/{len(testset)}] 残差范围:  min={residual.min().item():.4f}, max={residual.max().item():.4f}")
#             print(f"[{i}/{len(testset)}] 输出范围:  min={out.min().item():.4f}, max={out.max().item():.4f}")
#
#             # 6. 构造保存路径
#             filename = os.path.splitext(os.path.basename(name[0]))[0]
#             save_path = os.path.join(config.output_pth, f"{filename}-output.png")
#
#             # 7. 保存图像
#             torchvision.utils.save_image(out, save_path)
#             print(f"[{i}/{len(testset)}] 已保存：{save_path}")
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--snapshot_pth",
#         type=str,
#         default="snapshots/model_epoch_91.ckpt",
#         help="path to saved state_dict (.ckpt)",
#     )
#     parser.add_argument(
#         "--test_pth",
#         type=str,
#         default="dataset/t/",
#         help="directory of test images",
#     )
#     parser.add_argument(
#         "--batch_size", type=int, default=1, help="inference batch size"
#     )
#     parser.add_argument(
#         "--output_pth",
#         type=str,
#         default="results/",
#         help="directory to save output images",
#     )
#
#     config = parser.parse_args()
#     test(config)

# import torch
# import torch.nn.functional as F
# import torchvision
# from torchvision import transforms
# import os
# import argparse
#
# from model import UResNet_P
# from dataloader import TestDataSet
#
# def test(config):
#     # 1. 设备
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     # 2. 加载模型
#     model = UResNet_P().to(device)
#     ckpt = torch.load(config.snapshot_pth, map_location=device)
#     if 'state_dict' in ckpt:
#         ckpt = ckpt['state_dict']
#     model.load_state_dict(ckpt)
#     model.eval()
#
#     # 3. 输出目录
#     os.makedirs(config.output_pth, exist_ok=True)
#
#     # 4. 数据加载（同训练）
#     transform = transforms.Compose([
#         transforms.ToTensor()
#     ])
#     testset = TestDataSet(config.test_pth, transform)
#     print(f"共加载测试图片数量：{len(testset)}")
#     loader = torch.utils.data.DataLoader(
#         testset, batch_size=config.batch_size, shuffle=False
#     )
#
#     # 5. 准备锐化卷积核（Unsharp Mask 简版）
#     #   3×3 kernel: 中心 +5，其它 -1，groups=3 保证对每个通道独立处理
#     kernel = torch.tensor(
#         [[0, -1,  0],
#          [-1, 5, -1],
#          [0, -1,  0]],
#         dtype=torch.float32, device=device
#     ).unsqueeze(0).unsqueeze(0)  # shape [1,1,3,3]
#     kernel = kernel.repeat(3, 1, 1, 1)  # shape [3,1,3,3]
#
#     # 6. 推理 + 保存
#     for i, (img, names) in enumerate(loader, 1):
#         img = img.to(device)
#         with torch.no_grad():
#             residual, _ = model(img)
#             enhanced = img + residual
#             out = torch.clamp(enhanced, 0, 1)
#
#             # 打印范围
#             print(f"[{i}/{len(testset)}] in:    {img.min():.4f}–{img.max():.4f}, "
#                   f"res: {residual.min():.4f}–{residual.max():.4f}, "
#                   f"out: {out.min():.4f}–{out.max():.4f}")
#
#             # 7. 后处理锐化
#             #    padding=1 保持尺寸不变，groups=3 对每个通道做独立卷积
#             out = F.conv2d(out, kernel, padding=1, groups=3)
#             out = torch.clamp(out, 0, 1)
#
#         # 8. 保存
#         filename = os.path.splitext(os.path.basename(names[0]))[0]
#         save_path = os.path.join(config.output_pth, f"{filename}-output.png")
#         torchvision.utils.save_image(out, save_path)
#         print(f"[{i}/{len(testset)}] 已保存：{save_path}")
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--snapshot_pth",
#         type=str,
#         default="snapshots/model_epoch_91.ckpt",
#         help="path to saved state_dict (.ckpt)"
#     )
#     parser.add_argument(
#         "--test_pth",
#         type=str,
#         default="dataset/t/",
#         help="directory of test images"
#     )
#     parser.add_argument(
#         "--batch_size",
#         type=int, default=1, help="inference batch size"
#     )
#     parser.add_argument(
#         "--output_pth",
#         type=str, default="results/", help="directory to save output images"
#     )
#     config = parser.parse_args()
#     test(config)



# import torch
# import torchvision
# from torchvision import transforms
# import os
# import argparse
#
# from model import UResNet_P
# from dataloader import TestDataSet
#
# def test(config):
#     # 1. 设备配置
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     # 2. 实例化网络并加载 state_dict
#     model = UResNet_P().to(device)
#     ckpt = torch.load(config.snapshot_pth, map_location=device)
#     if 'state_dict' in ckpt:   # 支持 Lightning .ckpt 格式
#         ckpt = ckpt['state_dict']
#     model.load_state_dict(ckpt)
#     model.eval()
#
#     # 3. 创建输出目录
#     os.makedirs(config.output_pth, exist_ok=True)
#
#     # 4. 数据预处理与加载
#     transform = transforms.Compose([
#         transforms.Resize((config.resize, config.resize)),
#         transforms.ToTensor()
#     ])
#     testset = TestDataSet(config.test_pth, transform)
#     print(f"共加载测试图片数量：{len(testset)}")
#
#     loader = torch.utils.data.DataLoader(
#         testset,
#         batch_size=config.batch_size,
#         shuffle=False
#     )
#
#     # 5. 推理并保存
#     for i, (img, names) in enumerate(loader, 1):
#         img = img.to(device)
#         with torch.no_grad():
#             # forward 返回第一个张量就是增强后图，第二个是 edge_map
#             enhanced, _ = model(img)
#
#             # 视频里已经做了 sigmoid，这里只做 clamp 保证 [0,1]
#             out = torch.clamp(enhanced, 0, 1)
#
#             print(
#                 f"[{i}/{len(testset)}] 输出范围: "
#                 f"min={out.min().item():.4f}, max={out.max().item():.4f}"
#             )
#
#         # 构造并创建保存路径
#         filename  = os.path.splitext(os.path.basename(names[0]))[0]
#         save_path = os.path.join(config.output_pth, f"{filename}-output.png")
#
#         # 保存
#         torchvision.utils.save_image(out, save_path)
#         print(f"[{i}/{len(testset)}] 已保存：{save_path}")
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--snapshot_pth",
#         type=str,
#         default="snapshots/model_epoch_130.ckpt",
#         help="path to saved state_dict (.ckpt)"
#     )
#     parser.add_argument(
#         "--test_pth",
#         type=str,
#         default="dataset/t/",
#         help="directory of test images"
#     )
#     parser.add_argument(
#         "--batch_size",
#         type=int,
#         default=1,
#         help="inference batch size"
#     )
#     parser.add_argument(
#         "--resize",
#         type=int,
#         default=256,
#         help="将输入缩放到该尺寸再送入网络"
#     )
#     parser.add_argument(
#         "--output_pth",
#         type=str,
#         default="results/",
#         help="保存输出图像的目录"
#     )
#     config = parser.parse_args()
#     test(config)



import os
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from torch.nn import functional as F

from model import UResNet_P, Edge_Detector
from dataloader import TrainDataSet

# 保持 cudnn 高效
torch.backends.cudnn.benchmark = True

def train(config):
    device = torch.device(f"cuda:{config.cuda_id}" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 模型和边缘检测器
    model = UResNet_P().to(device)
    edge_detector = Edge_Detector().to(device)
    for p in edge_detector.parameters():
        p.requires_grad = False

    # 损失与优化器
    criterion = nn.MSELoss().to(device) if config.loss_type == 'MSE' else nn.L1Loss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.decay_rate)

    # 数据增广 + 转换
    tsfms = transforms.Compose([
        transforms.Resize((config.resize, config.resize)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor()
    ])

    # 训练/验证集加载
    train_ds = TrainDataSet(config.input_images_path, config.label_images_path, tsfms)
    val_ds   = TrainDataSet(config.val_input_path,     config.val_label_path,     tsfms)

    if config.smoke_test:
        subset = min(config.smoke_size, len(train_ds))
        train_ds = Subset(train_ds, list(range(subset)))
        print(f"Smoke test: using {subset} samples")

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True,
                              num_workers=config.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=config.batch_size, shuffle=False,
                              num_workers=config.num_workers, pin_memory=True)

    # 迭代
    for epoch in range(1, config.num_epochs+1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.num_epochs}")
        for inp, gt in pbar:
            inp, gt = inp.to(device), gt.to(device)

            optimizer.zero_grad()
            pred, edge_map = model(inp)
            # 全尺度图像损失
            img_loss = criterion(pred, gt)

            # 多尺度边缘损失
            # 全分辨率
            edge_full = criterion(edge_map, edge_detector(gt))
            # 半分辨率
            gt_h = F.interpolate(gt, scale_factor=0.5, mode='bilinear', align_corners=False)
            em_h = F.interpolate(edge_map, size=gt_h.shape[2:], mode='bilinear', align_corners=False)
            edge_h = criterion(em_h, edge_detector(gt_h))
            # 四分之一分辨率
            gt_q = F.interpolate(gt, scale_factor=0.25, mode='bilinear', align_corners=False)
            em_q = F.interpolate(edge_map, size=gt_q.shape[2:], mode='bilinear', align_corners=False)
            edge_q = criterion(em_q, edge_detector(gt_q))

            edge_loss = edge_full + 0.5 * edge_h + 0.25 * edge_q

            # 总损失
            if config.train_mode in ['P-S','P-A']:
                loss = img_loss + config.edge_weight * edge_loss
            else:
                loss = img_loss

            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()

        # 验证
        model.eval()
        psnr_vals = []
        with torch.no_grad():
            for inp, gt in val_loader:
                inp, gt = inp.to(device), gt.to(device)
                pred, _ = model(inp)
                for b in range(pred.size(0)):
                    pr = pred[b].cpu().permute(1,2,0).numpy()
                    gr = gt[b].cpu().permute(1,2,0).numpy()
                    psnr_vals.append(10 * np.log10(1.0 / ((pr-gr)**2).mean()))
        print(f"[Epoch {epoch}] Val PSNR: {np.mean(psnr_vals):.2f} dB")

        # 保存
        if epoch % config.snapshot_freq == 0:
            os.makedirs(config.snapshots_folder, exist_ok=True)
            path = os.path.join(config.snapshots_folder, f"model_epoch_{epoch}.ckpt")
            torch.save(model.state_dict(), path)
            print("Saved:", path)

    print("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 路径
    parser.add_argument('--input_images_path', type=str, default='./dataset/train/input/')
    parser.add_argument('--label_images_path', type=str, default='./dataset/train/label/')
    parser.add_argument('--val_input_path',   type=str, default='./dataset/val/input/')
    parser.add_argument('--val_label_path',   type=str, default='./dataset/val/label/')
    # 超参
    parser.add_argument('--lr',           type=float, default=2e-4)
    parser.add_argument('--decay_rate',   type=float, default=0.8)
    parser.add_argument('--step_size',    type=int,   default=50)
    parser.add_argument('--num_epochs',   type=int,   default=500)
    parser.add_argument('--loss_type',    type=str,   default='MSE', choices=['MSE','L1'])
    parser.add_argument('--train_mode',   type=str,   default='P-A', choices=['N','P-S','P-A'])
    parser.add_argument('--edge_weight',  type=float, default=1.2)
    # 其它
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