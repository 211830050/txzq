# # -*- coding: utf-8 -*-
# import os
# import argparse
# import numpy as np
# from tqdm import tqdm
#
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.optim.lr_scheduler import ReduceLROnPlateau
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
#     mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1,3,1,1)
#     std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1,3,1,1)
#     return (x - mean) / std
#
# class AddGaussianNoise:
#     def __init__(self, std=0.01): self.std = std
#     def __call__(self, img): return torch.clamp(img + torch.randn_like(img)*self.std, 0, 1)
#
# class RandomBlur:
#     def __init__(self, kernel_size=5, sigma=(0.1,2.0)):
#         self.ks, self.sigma = kernel_size, sigma
#     def __call__(self, img):
#         if np.random.rand() < 0.5:
#             s = np.random.uniform(*self.sigma)
#             return transforms.functional.gaussian_blur(img, self.ks, [s, s])
#         return img
#
# # --------------------------------------------------------------------------
# def train(config):
#     device = torch.device(f"cuda:{config.cuda_id}" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")
#
#     # 1. model + loss
#     model   = UNetPlus(base_channels=config.base_channels).to(device)
#     loss_fn = MultiScaleLoss(device)
#
#     # 2. perceptual net
#     vgg = None
#     if config.percep_weight > 0:
#         w = VGG19_Weights.IMAGENET1K_V1
#         vgg = vgg19(weights=w).features[:config.percep_layer].to(device).eval()
#         for p in vgg.parameters(): p.requires_grad = False
#
#     # 3. optimizer + scheduler
#     optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
#     scheduler = ReduceLROnPlateau(
#         optimizer,
#         mode='max',
#         factor=config.lr_factor,
#         patience=config.lr_patience,
#         verbose=True
#     )
#
#     # 4. AMP scaler
#     scaler = GradScaler(enabled=config.use_amp)
#
#     # 5. data transforms & loaders
#     light_tfms = transforms.Compose([
#         transforms.Resize((config.resize, config.resize)),
#         transforms.RandomHorizontalFlip(0.5),
#         transforms.ToTensor(),
#     ])
#     heavy_tfms = transforms.Compose([
#         transforms.RandomResizedCrop(config.resize, scale=(0.9,1.0)),
#         transforms.RandomHorizontalFlip(0.5),
#         transforms.RandomVerticalFlip(0.5),
#         transforms.RandomRotation(config.rotation),
#         transforms.ColorJitter(brightness=config.jitter, contrast=config.jitter),
#         RandomBlur(kernel_size=5, sigma=(0.1,2.0)),
#         transforms.ToTensor(),
#         AddGaussianNoise(std=config.noise_std),
#     ])
#
#     ds1 = TrainDataSet(config.input_images_path, config.label_images_path, light_tfms)
#     ds2 = TrainDataSet(config.input_images_path, config.label_images_path, heavy_tfms)
#     ds_val = TrainDataSet(
#         config.val_input_path, config.val_label_path,
#         transforms.Compose([
#             transforms.Resize((config.resize, config.resize)),
#             transforms.ToTensor(),
#         ])
#     )
#     if config.smoke_test:
#         n = min(config.smoke_size, len(ds1))
#         ds1 = Subset(ds1, list(range(n)))
#         ds2 = Subset(ds2, list(range(n)))
#
#     loader1 = DataLoader(
#         ds1, batch_size=config.batch_size, shuffle=True,
#         num_workers=config.num_workers, pin_memory=True,
#         persistent_workers=True, prefetch_factor=2, drop_last=True
#     )
#     loader2 = DataLoader(
#         ds2, batch_size=config.batch_size, shuffle=True,
#         num_workers=config.num_workers, pin_memory=True,
#         persistent_workers=True, prefetch_factor=2, drop_last=True
#     )
#     loader_val = DataLoader(
#         ds_val, batch_size=config.batch_size, shuffle=False,
#         num_workers=config.num_workers, pin_memory=True
#     )
#
#     best_psnr = 0.0
#     prev_loss = None
#
#     for epoch in range(1, config.num_epochs+1):
#         model.train()
#         loader = loader1 if epoch <= config.stage1_epochs else loader2
#         pbar = tqdm(loader, desc=f"Epoch [{epoch}/{config.num_epochs}]")
#
#         # smooth loss ramp
#         if epoch <= config.stage1_epochs:
#             alpha = 0.0
#         else:
#             alpha = min(1.0, (epoch - config.stage1_epochs) / config.ramp_epochs)
#
#         # optional LR drop at switch
#         if epoch == config.stage1_epochs + 1:
#             for pg in optimizer.param_groups:
#                 pg['lr'] *= config.switch_lr_factor
#
#         for inp, gt in pbar:
#             inp, gt = inp.to(device), gt.to(device)
#             optimizer.zero_grad()
#             with autocast(enabled=config.use_amp):
#                 pred = model(inp)
#                 # stage1 losses
#                 if epoch <= config.stage1_epochs:
#                     loss_c = torch.mean(torch.sqrt((pred-gt)**2 + 1e-6))
#                     loss_g = loss_fn.gradient_loss(pred, gt)
#                     base_loss = config.stage1_char_weight * loss_c + config.stage1_grad_weight * loss_g
#                     total = base_loss
#                 else:
#                     # multi-scale base
#                     losses = loss_fn(pred, gt, None)
#                     # SSIM
#                     loss_s = 1 - ms_ssim(pred, gt, data_range=1.0, size_average=True)
#                     # perceptual
#                     if vgg:
#                         pf = vgg(normalize_for_vgg(pred))
#                         gf = vgg(normalize_for_vgg(gt))
#                         loss_p = nn.L1Loss()(pf, gf)
#                     else:
#                         loss_p = 0.0
#                     # tv
#                     loss_tv = torch.mean(torch.abs(pred[:,:,1:,:]-pred[:,:,:-1,:])) + \
#                               torch.mean(torch.abs(pred[:,:,:,1:]-pred[:,:,:,:-1]))
#                     new_loss = (losses['total']
#                                 + config.ssim_weight * loss_s
#                                 + config.percep_weight * loss_p
#                                 + config.tv_weight * loss_tv)
#                     # smooth combination
#                     total = (1-alpha)*losses['total'] + alpha*new_loss
#
#                 # skip extreme
#                 if prev_loss and total.item() > 3 * prev_loss:
#                     continue
#                 prev_loss = total.item()
#
#             scaler.scale(total).backward()
#             scaler.unscale_(optimizer)
#             nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
#             scaler.step(optimizer)
#             scaler.update()
#             pbar.set_postfix(loss=f"{total.item():.4f}")
#
#         # validation
#         model.eval()
#         psnr_list = []
#         with torch.no_grad():
#             for inp, gt in loader_val:
#                 inp, gt = inp.to(device), gt.to(device)
#                 out = torch.clamp(model(inp), 0, 1)
#                 mse = F.mse_loss(out, gt)
#                 psnr_list.append(10 * np.log10(1.0 / mse.item()))
#         avg_psnr = np.mean(psnr_list)
#         print(f"Val PSNR: {avg_psnr:.2f} dB")
#
#         scheduler.step(avg_psnr)
#         if avg_psnr > best_psnr:
#             best_psnr = avg_psnr
#             torch.save(model.state_dict(), os.path.join(config.snapshots_folder, 'best.pth'))
#
#     print("Training complete.")
#
# if __name__=='__main__':
#     p = argparse.ArgumentParser()
#     p.add_argument('--cuda_id', type=int, default=0)
#     p.add_argument('--input_images_path', type=str, default='./dataset/train/input/')
#     p.add_argument('--label_images_path', type=str, default='./dataset/train/label/')
#     p.add_argument('--val_input_path', type=str, default='./dataset/val/input/')
#     p.add_argument('--val_label_path', type=str, default='./dataset/val/label/')
#     p.add_argument('--base_channels', type=int, default=64)
#     p.add_argument('--batch_size', type=int, default=8)
#     p.add_argument('--num_epochs', type=int, default=260)
#     p.add_argument('--resize', type=int, default=256)
#     p.add_argument('--stage1_epochs', type=int, default=50)
#     p.add_argument('--ramp_epochs', type=int, default=10, help='Number of epochs to ramp in new losses')
#     p.add_argument('--switch_lr_factor', type=float, default=0.1, help='LR multiplier at stage switch')
#     p.add_argument('--stage1_char_weight', type=float, default=1.0)
#     p.add_argument('--stage1_grad_weight', type=float, default=0.1)
#     p.add_argument('--rotation', type=int, default=15)
#     p.add_argument('--jitter', type=float, default=0.1)
#     p.add_argument('--noise_std', type=float, default=0.02)
#     p.add_argument('--lr', type=float, default=1e-4)
#     p.add_argument('--weight_decay', type=float, default=1e-5)
#     p.add_argument('--percep_weight', type=float, default=0.1)
#     p.add_argument('--percep_layer', type=int, default=16)
#     p.add_argument('--ssim_weight', type=float, default=0.5)
#     p.add_argument('--tv_weight', type=float, default=0.01)
#     p.add_argument('--grad_clip', type=float, default=0.5)
#     p.add_argument('--use_amp', action='store_true')
#     p.add_argument('--num_workers', type=int, default=8)
#     p.add_argument('--lr_factor', type=float, default=0.5)
#     p.add_argument('--lr_patience', type=int, default=5)
#     p.add_argument('--snapshots_folder', type=str, default='./snapshots/')
#     p.add_argument('--smoke_test', action='store_true')
#     p.add_argument('--smoke_size', type=int, default=100)
#     config = p.parse_args()
#     os.makedirs(config.snapshots_folder, exist_ok=True)
#     train(config)



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
    def __init__(self, std=0.01): self.std = std
    def __call__(self, img): return torch.clamp(img + torch.randn_like(img)*self.std, 0, 1)

class RandomBlur:
    def __init__(self, kernel_size=5, sigma=(0.1,2.0)):
        self.ks, self.sigma = kernel_size, sigma
    def __call__(self, img):
        if np.random.rand() < 0.5:
            s = np.random.uniform(*self.sigma)
            return transforms.functional.gaussian_blur(img, self.ks, [s, s])
        return img

# --------------------------------------------------------------------------
def train(config):
    device = torch.device(f"cuda:{config.cuda_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. model + loss
    model   = UNetPlus(base_channels=config.base_channels).to(device)
    loss_fn = MultiScaleLoss(device)

    # 2. perceptual net
    vgg = None
    if config.percep_weight > 0:
        w = VGG19_Weights.IMAGENET1K_V1
        vgg = vgg19(weights=w).features[:config.percep_layer].to(device).eval()
        for p in vgg.parameters(): p.requires_grad = False

    # 3. optimizer + scheduler
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=config.lr_factor,
        patience=config.lr_patience,
        verbose=True
    )

    # 4. AMP scaler
    scaler = GradScaler(enabled=config.use_amp)

    # 5. data transforms & loaders
    light_tfms = transforms.Compose([
        transforms.Resize((config.resize, config.resize)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
    ])
    heavy_tfms = transforms.Compose([
        transforms.RandomResizedCrop(config.resize, scale=(0.9,1.0)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomRotation(config.rotation),
        transforms.ColorJitter(brightness=config.jitter, contrast=config.jitter),
        RandomBlur(kernel_size=5, sigma=(0.1,2.0)),
        transforms.ToTensor(),
        AddGaussianNoise(std=config.noise_std),
    ])

    ds1 = TrainDataSet(config.input_images_path, config.label_images_path, light_tfms)
    ds2 = TrainDataSet(config.input_images_path, config.label_images_path, heavy_tfms)
    ds_val = TrainDataSet(
        config.val_input_path, config.val_label_path,
        transforms.Compose([
            transforms.Resize((config.resize, config.resize)),
            transforms.ToTensor(),
        ])
    )
    if config.smoke_test:
        n = min(config.smoke_size, len(ds1))
        ds1 = Subset(ds1, list(range(n)))
        ds2 = Subset(ds2, list(range(n)))

    loader1 = DataLoader(
        ds1, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=True,
        persistent_workers=True, prefetch_factor=2, drop_last=True
    )
    loader2 = DataLoader(
        ds2, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=True,
        persistent_workers=True, prefetch_factor=2, drop_last=True
    )
    loader_val = DataLoader(
        ds_val, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=True
    )

    best_psnr = 0.0
    prev_loss = None

    for epoch in range(1, config.num_epochs+1):
        model.train()
        loader = loader1 if epoch <= config.stage1_epochs else loader2
        pbar = tqdm(loader, desc=f"Epoch [{epoch}/{config.num_epochs}]")

        # smooth loss ramp
        if epoch <= config.stage1_epochs:
            alpha = 0.0
        else:
            alpha = min(1.0, (epoch - config.stage1_epochs) / config.ramp_epochs)

        # optional LR drop at switch
        if epoch == config.stage1_epochs + 1:
            for pg in optimizer.param_groups:
                pg['lr'] *= config.switch_lr_factor

        for inp, gt in pbar:
            inp, gt = inp.to(device), gt.to(device)
            optimizer.zero_grad()
            with autocast(enabled=config.use_amp):
                pred = model(inp)
                # stage1 losses
                if epoch <= config.stage1_epochs:
                    loss_c = torch.mean(torch.sqrt((pred-gt)**2 + 1e-6))
                    loss_g = loss_fn.gradient_loss(pred, gt)
                    base_loss = config.stage1_char_weight * loss_c + config.stage1_grad_weight * loss_g
                    total = base_loss
                else:
                    # multi-scale base
                    losses = loss_fn(pred, gt, None)
                    # SSIM
                    loss_s = 1 - ms_ssim(pred, gt, data_range=1.0, size_average=True)
                    # perceptual
                    if vgg:
                        pf = vgg(normalize_for_vgg(pred))
                        gf = vgg(normalize_for_vgg(gt))
                        loss_p = nn.L1Loss()(pf, gf)
                    else:
                        loss_p = 0.0
                    # tv
                    loss_tv = torch.mean(torch.abs(pred[:,:,1:,:]-pred[:,:,:-1,:])) + \
                              torch.mean(torch.abs(pred[:,:,:,1:]-pred[:,:,:,:-1]))
                    new_loss = (losses['total']
                                + config.ssim_weight * loss_s
                                + config.percep_weight * loss_p
                                + config.tv_weight * loss_tv)
                    # smooth combination
                    total = (1-alpha)*losses['total'] + alpha*new_loss

                # skip extreme
                if prev_loss and total.item() > 3 * prev_loss:
                    continue
                prev_loss = total.item()

            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            pbar.set_postfix(loss=f"{total.item():.4f}")

        # validation
        model.eval()
        psnr_list = []
        with torch.no_grad():
            for inp, gt in loader_val:
                inp, gt = inp.to(device), gt.to(device)
                out = torch.clamp(model(inp), 0, 1)
                mse = F.mse_loss(out, gt)
                psnr_list.append(10 * np.log10(1.0 / mse.item()))
        avg_psnr = np.mean(psnr_list)
        print(f"Val PSNR: {avg_psnr:.2f} dB")

        scheduler.step(avg_psnr)
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            torch.save(model.state_dict(), os.path.join(config.snapshots_folder, 'best.pth'))

    print("Training complete.")

if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--cuda_id', type=int, default=0)
    p.add_argument('--input_images_path', type=str, default='./dataset/train/input/')
    p.add_argument('--label_images_path', type=str, default='./dataset/train/label/')
    p.add_argument('--val_input_path', type=str, default='./dataset/val/input/')
    p.add_argument('--val_label_path', type=str, default='./dataset/val/label/')
    p.add_argument('--base_channels', type=int, default=64)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--num_epochs', type=int, default=280)
    p.add_argument('--resize', type=int, default=256)
    p.add_argument('--stage1_epochs', type=int, default=50)
    p.add_argument('--ramp_epochs', type=int, default=10, help='Number of epochs to ramp in new losses')
    p.add_argument('--switch_lr_factor', type=float, default=0.1, help='LR multiplier at stage switch')
    p.add_argument('--stage1_char_weight', type=float, default=1.0)
    p.add_argument('--stage1_grad_weight', type=float, default=0.1)
    p.add_argument('--rotation', type=int, default=15)
    p.add_argument('--jitter', type=float, default=0.1)
    p.add_argument('--noise_std', type=float, default=0.02)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=1e-5)
    p.add_argument('--percep_weight', type=float, default=0.1)
    p.add_argument('--percep_layer', type=int, default=16)
    p.add_argument('--ssim_weight', type=float, default=0.5)
    p.add_argument('--tv_weight', type=float, default=0.01)
    p.add_argument('--grad_clip', type=float, default=0.5)
    p.add_argument('--use_amp', action='store_true')
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--lr_factor', type=float, default=0.5)
    p.add_argument('--lr_patience', type=int, default=5)
    p.add_argument('--snapshots_folder', type=str, default='./snapshots/')
    p.add_argument('--snapshot_freq', type=int, default=50)
    p.add_argument('--smoke_test', action='store_true')
    p.add_argument('--smoke_size', type=int, default=100)
    config = p.parse_args()
    os.makedirs(config.snapshots_folder, exist_ok=True)
    train(config)
