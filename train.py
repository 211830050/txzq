# # -*- coding: utf-8 -*-
# import os
# import argparse
# import numpy as np
# from tqdm import tqdm
#
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
# from torch.utils.data import DataLoader, Subset
# from torchvision import transforms
# from torch.nn import functional as F
# from torch.cuda.amp import autocast, GradScaler
#
# from pytorch_msssim import ms_ssim
# from torchvision.models import vgg19, VGG19_Weights
#
# from model import UNetPlus, MultiScaleLoss
# from dataloader import TrainDataSet
#
# torch.backends.cudnn.benchmark = True
#
# def normalize_for_vgg(x):
#     """Normalize tensor for VGG input"""
#     mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1,3,1,1)
#     std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1,3,1,1)
#     return (x - mean) / std
#
# class AddGaussianNoise:
#     def __init__(self, mean=0.0, std=0.01):
#         self.mean = mean
#         self.std = std
#     def __call__(self, tensor):
#         return torch.clamp(tensor + torch.randn_like(tensor) * self.std + self.mean, 0.0, 1.0)
#
# class RandomBlur:
#     def __init__(self, kernel_size=5, sigma=(0.1,2.0)):
#         self.kernel_size = kernel_size
#         self.sigma = sigma
#     def __call__(self, img):
#         if np.random.rand() < 0.5:
#             sigma = np.random.uniform(self.sigma[0], self.sigma[1])
#             return transforms.functional.gaussian_blur(img, self.kernel_size, sigma)
#         return img
#
#
# def train(config):
#     device = torch.device(f"cuda:{config.cuda_id}" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")
#
#     # model and loss
#     model = UNetPlus(base_channels=config.base_channels).to(device)
#     loss_fn = MultiScaleLoss(device)
#
#     # perceptual net
#     vgg = None
#     if config.percep_weight > 0:
#         vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:config.percep_layer].to(device).eval()
#         for p in vgg.parameters(): p.requires_grad = False
#
#     optimizer = optim.AdamW(
#         model.parameters(), lr=config.lr, weight_decay=config.weight_decay
#     )
#     scheduler_plateau = ReduceLROnPlateau(
#         optimizer, mode='max', factor=0.5, patience=5, verbose=True
#     )
#     scheduler_cosine = CosineAnnealingLR(
#         optimizer, T_max=config.num_epochs - config.stage1_epochs, eta_min=config.lr * 1e-2
#     )
#
#     scaler = GradScaler(enabled=config.use_amp)
#
#     # transforms
#     light_tfms = transforms.Compose([
#         transforms.Resize((config.resize,config.resize)),
#         transforms.RandomHorizontalFlip(0.5),
#         transforms.ToTensor()
#     ])
#     heavy_tfms = transforms.Compose([
#         transforms.RandomResizedCrop(config.resize, scale=(0.8,1.0)),
#         transforms.RandomHorizontalFlip(0.5),
#         transforms.RandomVerticalFlip(0.5),
#         transforms.RandomRotation(config.rotation),
#         transforms.ColorJitter(brightness=config.jitter, contrast=config.jitter),
#         RandomBlur(kernel_size=5, sigma=(0.1,2.0)),
#         transforms.ToTensor(),
#         AddGaussianNoise(std=config.noise_std)
#     ])
#
#     ds_light = TrainDataSet(config.input_images_path, config.label_images_path, light_tfms)
#     ds_heavy = TrainDataSet(config.input_images_path, config.label_images_path, heavy_tfms)
#     ds_val = TrainDataSet(config.val_input_path, config.val_label_path,
#                           transforms.Compose([transforms.Resize((config.resize,config.resize)), transforms.ToTensor()]))
#     if config.smoke_test:
#         limit = min(config.smoke_size, len(ds_light))
#         ds_light = Subset(ds_light, range(limit))
#         ds_heavy = Subset(ds_heavy, range(limit))
#     loader_light = DataLoader(
#         ds_light, batch_size=config.batch_size, shuffle=True,
#         num_workers=config.num_workers, pin_memory=True,
#         persistent_workers=True, prefetch_factor=2, drop_last=True
#     )
#     loader_heavy = DataLoader(
#         ds_heavy, batch_size=config.batch_size, shuffle=True,
#         num_workers=config.num_workers, pin_memory=True,
#         persistent_workers=True, prefetch_factor=2, drop_last=True
#     )
#     loader_val = DataLoader(
#         ds_val, batch_size=config.batch_size, shuffle=False,
#         num_workers=config.num_workers, pin_memory=True
#     )
#
#     best_psnr=0.0
#     for epoch in range(1, config.num_epochs+1):
#         model.train()
#         loader = loader_light if epoch <= config.stage1_epochs else loader_heavy
#         pbar = tqdm(loader, desc=f"Epoch [{epoch}/{config.num_epochs}]")
#         for inp, gt in pbar:
#             inp, gt = inp.to(device), gt.to(device)
#             optimizer.zero_grad()
#             with autocast(enabled=config.use_amp):
#                 pred = model(inp)
#                 # perceptual features
#                 if vgg:
#                     gt_feat = vgg(normalize_for_vgg(gt))
#                     pr_feat = vgg(normalize_for_vgg(pred))
#                 # loss
#                 if epoch <= config.stage1_epochs:
#                     # Stage 1: only Charbonnier + gradient loss
#                     # Charbonnier loss (robust L1)
#                     char_loss = torch.mean(torch.sqrt((pred - gt)**2 + 1e-6))
#                     # Gradient consistency
#                     grad_loss = loss_fn.gradient_loss(pred, gt)
#                     total = config.stage1_char_weight * char_loss + config.stage1_grad_weight * grad_loss
#                 else:
#                     losses = loss_fn(pred, gt, (pr_feat, gt_feat))
#                     total = losses['total']
#             scaler.scale(total).backward()
#             scaler.unscale_(optimizer)
#             nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
#             scaler.step(optimizer)
#             scaler.update()
#             pbar.set_postfix(loss=f"{total.item():.4f}")
#
#         # scheduler
#         if epoch <= config.stage1_epochs:
#             # plateau on PSNR
#             model.eval(); ps=[10*np.log10(1.0/F.mse_loss(model(x.to(device)), y.to(device)).item()) for x,y in loader_val]
#             scheduler_plateau.step(np.mean(ps))
#         else:
#             scheduler_cosine.step()
#
#         # validation
#         model.eval()
#         psnr_vals=[]
#         with torch.no_grad():
#             for inp, gt in loader_val:
#                 inp, gt = inp.to(device), gt.to(device)
#                 pr = torch.clamp(model(inp),0,1)
#                 mse = F.mse_loss(pr,gt)
#                 psnr_vals.append(10*np.log10(1.0/mse.item()))
#         avg_psnr = np.mean(psnr_vals)
#         print(f"Val PSNR: {avg_psnr:.2f} dB")
#         if avg_psnr > best_psnr:
#             best_psnr = avg_psnr
#             torch.save(model.state_dict(), os.path.join(config.snapshots_folder,'best.pth'))
#
#     print("Training complete.")
#
# if __name__=='__main__':
#     p = argparse.ArgumentParser()
#     # CUDA device id
#     p.add_argument('--cuda_id', type=int, default=0, help='CUDA device index')
#     p.add_argument('--input_images_path',type=str,default='./dataset/train/input/')
#     p.add_argument('--label_images_path',type=str,default='./dataset/train/label/')
#     p.add_argument('--val_input_path',type=str,default='./dataset/val/input/')
#     p.add_argument('--val_label_path',type=str,default='./dataset/val/label/')
#     p.add_argument('--base_channels', type=int, default=64, help='Number of base channels in UNetPlus')
#     p.add_argument('--batch_size',type=int,default=8)
#     p.add_argument('--num_epochs',type=int,default=280)
#     p.add_argument('--resize',type=int,default=256)
#     p.add_argument('--stage1_epochs',type=int,default=50)
#     p.add_argument('--stage1_char_weight',type=float,default=1.0)
#     p.add_argument('--stage1_grad_weight',type=float,default=0.1)
#     p.add_argument('--rotation',type=int,default=15)
#     p.add_argument('--jitter',type=float,default=0.1)
#     p.add_argument('--noise_std',type=float,default=0.02)
#     p.add_argument('--lr',type=float,default=1e-4)
#     p.add_argument('--weight_decay',type=float,default=1e-5)
#     p.add_argument('--percep_weight',type=float,default=0.1)
#     p.add_argument('--percep_layer',type=int,default=16)
#     p.add_argument('--grad_clip',type=float,default=1.0)
#     p.add_argument('--use_amp',action='store_true')
#     p.add_argument('--num_workers',type=int,default=18)
#
#     p.add_argument('--snapshots_folder',type=str,default='./snapshots/')
#     p.add_argument('--smoke_test',action='store_true')
#     p.add_argument('--smoke_size',type=int,default=100)
#     p.add_argument('--snapshot_freq', type=int, default=28)
#     config=p.parse_args()
#     os.makedirs(config.snapshots_folder,exist_ok=True)
#     train(config)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#


# -*- coding: utf-8 -*-
import os
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torch.nn import functional as F
from torch.cuda.amp import autocast, GradScaler

from pytorch_msssim import ms_ssim
from torchvision.models import vgg19, VGG19_Weights

from model import UNetPlus, MultiScaleLoss
from dataloader import TrainDataSet

torch.backends.cudnn.benchmark = True

def normalize_for_vgg(x):
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1,3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1,3,1,1)
    return (x - mean) / std

class AddGaussianNoise:
    def __init__(self, mean=0.0, std=0.01):
        self.mean, self.std = mean, std
    def __call__(self, img):
        return torch.clamp(img + torch.randn_like(img)*self.std + self.mean, 0, 1)

class RandomBlur:
    def __init__(self, kernel_size=5, sigma=(0.1,2.0)):
        self.ks, self.sigma = kernel_size, sigma
    def __call__(self, img):
        if np.random.rand()<0.5:
            s = np.random.uniform(*self.sigma)
            return transforms.functional.gaussian_blur(img, self.ks, [s,s])
        return img

def train(config):
    device = torch.device(f"cuda:{config.cuda_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. model + loss
    model   = UNetPlus(base_channels=config.base_channels).to(device)
    loss_fn = MultiScaleLoss(device)

    # 2. perceptual net
    vgg = None
    if config.percep_weight>0:
        w = VGG19_Weights.IMAGENET1K_V1
        vgg = vgg19(weights=w).features[:config.percep_layer].to(device).eval()
        for p in vgg.parameters(): p.requires_grad=False

    # 3. optimizer + plateau scheduler
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=config.lr_factor,
        patience=config.lr_patience,
        verbose=True
    )

    # 4. AMP
    scaler = GradScaler(enabled=config.use_amp)

    # 5. transforms & loaders
    light_tfms = transforms.Compose([
        transforms.Resize((config.resize,config.resize)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor()
    ])
    heavy_tfms = transforms.Compose([
        transforms.RandomResizedCrop(config.resize, scale=(0.8,1.0)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomRotation(config.rotation),
        transforms.ColorJitter(brightness=config.jitter, contrast=config.jitter),
        RandomBlur(kernel_size=5, sigma=(0.1,2.0)),
        transforms.ToTensor(),
        AddGaussianNoise(std=config.noise_std)
    ])
    ds_light = TrainDataSet(config.input_images_path, config.label_images_path, light_tfms)
    ds_heavy = TrainDataSet(config.input_images_path, config.label_images_path, heavy_tfms)
    ds_val   = TrainDataSet(config.val_input_path,     config.val_label_path,
                            transforms.Compose([
                                transforms.Resize((config.resize,config.resize)),
                                transforms.ToTensor()
                            ]))
    if config.smoke_test:
        limit = min(config.smoke_size, len(ds_light))
        ds_light = Subset(ds_light, range(limit))
        ds_heavy = Subset(ds_heavy, range(limit))

    loader_light = DataLoader(
        ds_light, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=True,
        persistent_workers=True, prefetch_factor=2, drop_last=True
    )
    loader_heavy = DataLoader(
        ds_heavy, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=True,
        persistent_workers=True, prefetch_factor=2, drop_last=True
    )
    loader_val = DataLoader(
        ds_val,   batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=True
    )

    best_psnr = 0.0

    # 6. training loop
    for epoch in range(1, config.num_epochs+1):
        model.train()
        loader = loader_light if epoch<=config.stage1_epochs else loader_heavy
        pbar = tqdm(loader, desc=f"Epoch [{epoch}/{config.num_epochs}]")

        for inp, gt in pbar:
            inp, gt = inp.to(device), gt.to(device)
            optimizer.zero_grad()
            with autocast(enabled=config.use_amp):
                pred = model(inp)
                if vgg:
                    gt_feat = vgg(normalize_for_vgg(gt))
                    pr_feat = vgg(normalize_for_vgg(pred))
                if epoch<=config.stage1_epochs:
                    char_loss = torch.mean(torch.sqrt((pred-gt)**2 + 1e-6))
                    grad_loss = loss_fn.gradient_loss(pred, gt)
                    total = config.stage1_char_weight*char_loss + config.stage1_grad_weight*grad_loss
                else:
                    losses = loss_fn(pred, gt, (pr_feat, gt_feat))
                    total = losses['total']
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            pbar.set_postfix(loss=f"{total.item():.4f}")

        # 7. validation
        model.eval()
        psnr_vals = []
        with torch.no_grad():
            for inp, gt in loader_val:
                inp, gt = inp.to(device), gt.to(device)
                pr = torch.clamp(model(inp), 0, 1)
                mse = F.mse_loss(pr, gt)
                psnr_vals.append(10*np.log10(1.0/mse.item()))
        avg_psnr = np.mean(psnr_vals)
        print(f"Val PSNR: {avg_psnr:.2f} dB")

        # 8. scheduler step
        scheduler.step(avg_psnr)

        # 9. save best
        if avg_psnr>best_psnr:
            best_psnr = avg_psnr
            torch.save(
                model.state_dict(),
                os.path.join(config.snapshots_folder, 'best.pth')
            )

    print("Training complete.")

if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--cuda_id',           type=int,   default=0)
    p.add_argument('--input_images_path', type=str,   default='./dataset/train/input/')
    p.add_argument('--label_images_path', type=str,   default='./dataset/train/label/')
    p.add_argument('--val_input_path',    type=str,   default='./dataset/val/input/')
    p.add_argument('--val_label_path',    type=str,   default='./dataset/val/label/')
    p.add_argument('--base_channels',     type=int,   default=64)
    p.add_argument('--batch_size',        type=int,   default=8)
    p.add_argument('--num_epochs',        type=int,   default=260)
    p.add_argument('--resize',            type=int,   default=256)
    p.add_argument('--stage1_epochs',     type=int,   default=50)
    p.add_argument('--stage1_char_weight',type=float, default=1.0)
    p.add_argument('--stage1_grad_weight',type=float, default=0.1)
    p.add_argument('--rotation',          type=int,   default=15)
    p.add_argument('--jitter',            type=float, default=0.1)
    p.add_argument('--noise_std',         type=float, default=0.02)
    p.add_argument('--lr',                type=float, default=1e-4)
    p.add_argument('--weight_decay',      type=float, default=1e-5)
    p.add_argument('--percep_weight',     type=float, default=0.1)
    p.add_argument('--percep_layer',      type=int,   default=16)
    p.add_argument('--grad_clip',         type=float, default=1.0)
    p.add_argument('--use_amp',           action='store_true')
    p.add_argument('--num_workers',       type=int,   default=8)

    # 新增这两行，避免 AttributeError
    p.add_argument('--lr_factor',   type=float, default=0.5,
                   help='ReduceLROnPlateau 的衰减因子')
    p.add_argument('--lr_patience', type=int,   default=5,
                   help='ReduceLROnPlateau 的耐心轮数')

    p.add_argument('--snapshots_folder', type=str, default='./snapshots/')
    p.add_argument('--smoke_test',       action='store_true')
    p.add_argument('--smoke_size',       type=int,   default=100)
    config = p.parse_args()
    os.makedirs(config.snapshots_folder, exist_ok=True)
    train(config)