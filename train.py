#
#
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
# from torchvision import transforms
# from torch.utils.data import DataLoader, Subset
# from torch.nn import functional as F
#
# # SSIM
# from pytorch_msssim import ssim
# # Perceptual VGG
# from torchvision.models import vgg19
#
# from model import UResNet_P, Edge_Detector
# from dataloader import TrainDataSet
#
# torch.backends.cudnn.benchmark = True
#
# def train(config):
#     # Device
#     device = torch.device(f"cuda:{config.cuda_id}" if torch.cuda.is_available() else "cpu")
#     print("Using device:", device)
#
#     # Model + frozen edge detector
#     model = UResNet_P().to(device)
#     edge_detector = Edge_Detector().to(device)
#     for p in edge_detector.parameters(): p.requires_grad=False
#
#     # Perceptual network
#     perceptual_net = None
#     if config.percep_weight > 0:
#         perceptual_net = vgg19(pretrained=True).features[:config.percep_layer].to(device).eval()
#         for p in perceptual_net.parameters(): p.requires_grad=False
#
#     # Losses and optimizer
#     criterion = nn.MSELoss().to(device) if config.loss_type=='MSE' else nn.L1Loss().to(device)
#     optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
#     scheduler = ReduceLROnPlateau(optimizer,
#                                   mode='max',
#                                   factor=config.lr_factor,
#                                   patience=config.lr_patience,
#                                   verbose=True)
#
#     # Data transforms
#     tsfms_train = transforms.Compose([
#         transforms.RandomResizedCrop(config.resize, scale=(0.8,1.0)),
#         transforms.RandomHorizontalFlip(),
#         transforms.ColorJitter(brightness=0.1, contrast=0.1),
#         transforms.ToTensor()
#     ])
#     tsfms_val = transforms.Compose([
#         transforms.Resize((config.resize,config.resize)),
#         transforms.ToTensor()
#     ])
#
#     # Datasets and loaders
#     train_ds = TrainDataSet(config.input_images_path, config.label_images_path, tsfms_train)
#     val_ds   = TrainDataSet(config.val_input_path,     config.val_label_path,    tsfms_val)
#     if config.smoke_test:
#         train_ds = Subset(train_ds, list(range(min(config.smoke_size,len(train_ds)))))
#     train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True,
#                               num_workers=config.num_workers, pin_memory=True)
#     val_loader   = DataLoader(val_ds,   batch_size=config.batch_size, shuffle=False,
#                               num_workers=config.num_workers, pin_memory=True)
#
#     best_ssim = 0.0
#     no_improve = 0
#
#     for epoch in range(1, config.num_epochs+1):
#         model.train()
#         pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.num_epochs}")
#         for inp, gt in pbar:
#             inp,gt = inp.to(device), gt.to(device)
#             optimizer.zero_grad()
#             pred, edge_map = model(inp)
#
#             # MSE/L1
#             img_loss = criterion(pred, gt)
#             # Edge
#             edge_loss = criterion(edge_map, edge_detector(gt))
#             # SSIM
#             ssim_loss = 1 - ssim(pred, gt, data_range=1.0, size_average=True)
#             # Perceptual
#             if perceptual_net:
#                 feat_p = perceptual_net(pred)
#                 feat_g = perceptual_net(gt)
#                 percep_loss = nn.L1Loss()(feat_p, feat_g)
#             else:
#                 percep_loss = torch.tensor(0.0, device=device)
#             # TV Loss
#             tv_loss = torch.mean(torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])) + \
#                       torch.mean(torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:]))
#
#             # Combined loss
#             loss = img_loss \
#                    + config.edge_weight * edge_loss \
#                    + config.ssim_weight * ssim_loss \
#                    + config.percep_weight * percep_loss \
#                    + config.tv_weight * tv_loss
#
#             loss.backward()
#             nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
#             optimizer.step()
#
#             pbar.set_postfix({
#                 'total': f"{loss.item():.4f}",
#                 'img': f"{img_loss.item():.4f}",
#                 'edge': f"{edge_loss.item():.4f}",
#                 'ssim': f"{ssim_loss.item():.4f}",
#                 'tv': f"{tv_loss.item():.4f}" + (f", perc={percep_loss.item():.4f}" if perceptual_net else "")
#             })
#
#         # Validation
#         model.eval()
#         psnr_vals, ssim_vals = [], []
#         with torch.no_grad():
#             for inp, gt in val_loader:
#                 inp, gt = inp.to(device), gt.to(device)
#                 pred, _ = model(inp)
#                 pred = torch.clamp(pred, 0, 1)
#                 # per sample
#                 for b in range(pred.size(0)):
#                     pr = pred[b].cpu().permute(1,2,0).numpy()
#                     gr = gt[b].cpu().permute(1,2,0).numpy()
#                     mse = ((pr-gr)**2).mean()
#                     psnr_vals.append(10 * np.log10(1.0/mse) if mse>0 else float('inf'))
#                 ssim_vals.append(ssim(pred, gt, data_range=1.0, size_average=True).item())
#         avg_psnr = np.mean(psnr_vals)
#         avg_ssim = np.mean(ssim_vals)
#         print(f"[Epoch {epoch}] Val PSNR: {avg_psnr:.2f} dB, SSIM: {avg_ssim:.4f}")
#
#         # Scheduler step
#         scheduler.step(avg_ssim)
#
#         # Early stopping
#         if avg_ssim > best_ssim + config.es_delta:
#             best_ssim = avg_ssim
#             no_improve = 0
#         else:
#             no_improve += 1
#         if no_improve >= config.es_patience:
#             print(f"Early stopping at epoch {epoch}")
#             break
#
#         # Checkpoint
#         if epoch % config.snapshot_freq == 0:
#             os.makedirs(config.snapshots_folder, exist_ok=True)
#             ckpt_path = os.path.join(config.snapshots_folder, f"epoch{epoch}_psnr{avg_psnr:.2f}.ckpt")
#             torch.save(model.state_dict(), ckpt_path)
#             print("Saved:", ckpt_path)
#
#     print("Training complete.")
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     # paths
#     parser.add_argument('--input_images_path', type=str, default='./dataset/train/input/')
#     parser.add_argument('--label_images_path', type=str, default='./dataset/train/label/')
#     parser.add_argument('--val_input_path',   type=str, default='./dataset/val/input/')
#     parser.add_argument('--val_label_path',   type=str, default='./dataset/val/label/')
#     # hyperparams
#     parser.add_argument('--lr',            type=float, default=5e-4)
#     parser.add_argument('--weight_decay', type=float, default=1e-5)
#     parser.add_argument('--num_epochs',    type=int,   default=500)
#     parser.add_argument('--loss_type',     type=str,   default='MSE', choices=['MSE','L1'])
#     parser.add_argument('--edge_weight',   type=float, default=3.0)
#     parser.add_argument('--ssim_weight',   type=float, default=0.5)
#     parser.add_argument('--percep_weight', type=float, default=0.05)
#     parser.add_argument('--percep_layer',  type=int,   default=8)
#     parser.add_argument('--tv_weight',     type=float, default=0.01)
#     parser.add_argument('--grad_clip',     type=float, default=3.0)
#     # scheduler
#     parser.add_argument('--lr_factor',   type=float, default=0.5)
#     parser.add_argument('--lr_patience', type=int,   default=10)
#     # early stopping
#     parser.add_argument('--es_patience', type=int,   default=20)
#     parser.add_argument('--es_delta',    type=float, default=0.001)
#     # other
#     parser.add_argument('--batch_size',   type=int,   default=8)
#     parser.add_argument('--resize',       type=int,   default=256)
#     parser.add_argument('--cuda_id',      type=int,   default=0)
#     parser.add_argument('--snapshot_freq',type=int,   default=50)
#     parser.add_argument('--snapshots_folder', type=str, default='./snapshots/')
#     parser.add_argument('--num_workers',  type=int,   default=8)
#     parser.add_argument('--smoke_test',   action='store_true')
#     parser.add_argument('--smoke_size',   type=int,   default=500)
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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from torch.nn import functional as F

# SSIM
from pytorch_msssim import ssim
# Perceptual VGG
from torchvision.models import vgg19

from model import UResNet_P, Edge_Detector
from dataloader import TrainDataSet

torch.backends.cudnn.benchmark = True

class AddGaussianNoise:
    def __init__(self, mean=0.0, std=0.01):
        self.mean = mean
        self.std = std
    def __call__(self, tensor):
        return torch.clamp(tensor + torch.randn_like(tensor) * self.std + self.mean, 0.0, 1.0)

def train(config):
    # Device
    device = torch.device(f"cuda:{config.cuda_id}" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Model + frozen edge detector
    model = UResNet_P().to(device)
    edge_detector = Edge_Detector().to(device)
    for p in edge_detector.parameters(): p.requires_grad = False

    # Perceptual network
    perceptual_net = None
    if config.percep_weight > 0:
        perceptual_net = vgg19(weights=None).features[:config.percep_layer].to(device).eval()
        for p in perceptual_net.parameters(): p.requires_grad = False

    # Loss and optimizer
    criterion = nn.MSELoss().to(device) if config.loss_type == 'MSE' else nn.L1Loss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=config.lr_factor,
        patience=config.lr_patience,
        verbose=True
    )

    # Data transforms
    tsfms_train = transforms.Compose([
        transforms.RandomResizedCrop(config.resize, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(5),
        transforms.ColorJitter(brightness=config.jitter, contrast=config.jitter),
        transforms.ToTensor(),
        AddGaussianNoise(mean=0.0, std=config.noise_std)
    ])
    tsfms_val = transforms.Compose([
        transforms.Resize((config.resize, config.resize)),
        transforms.ToTensor()
    ])

    # Datasets
    train_ds = TrainDataSet(config.input_images_path, config.label_images_path, tsfms_train)
    val_ds   = TrainDataSet(config.val_input_path, config.val_label_path, tsfms_val)
    if config.smoke_test:
        train_ds = Subset(train_ds, list(range(min(config.smoke_size, len(train_ds)))))

    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=True
    )

    best_ssim = 0.0
    no_improve = 0

    for epoch in range(1, config.num_epochs + 1):
        # Warm-up learning rate
        if epoch <= config.warmup_epochs:
            lr = config.lr * epoch / config.warmup_epochs
            for pg in optimizer.param_groups:
                pg['lr'] = lr

        # Dynamic loss weights
        w_edge = config.edge_weight * (epoch / config.num_epochs)
        w_ssim = config.ssim_weight * (epoch / config.num_epochs)
        w_percep = config.percep_weight * (epoch / config.num_epochs)
        w_tv = config.tv_weight * (epoch / config.num_epochs)

        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.num_epochs}")
        for inp, gt in pbar:
            inp, gt = inp.to(device), gt.to(device)
            optimizer.zero_grad()
            pred, edge_map = model(inp)

            # Loss components
            img_loss = criterion(pred, gt)
            edge_loss = criterion(edge_map, edge_detector(gt))
            ssim_loss = 1 - ssim(pred, gt, data_range=1.0, size_average=True)
            if perceptual_net:
                feat_p = perceptual_net(pred)
                feat_g = perceptual_net(gt)
                percep_loss = nn.L1Loss()(feat_p, feat_g)
            else:
                percep_loss = torch.tensor(0.0, device=device)
            tv_loss = (
                torch.mean(torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])) +
                torch.mean(torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:]))
            )

            # Combined loss
            loss = (
                img_loss
                + w_edge * edge_loss
                + w_ssim * ssim_loss
                + w_percep * percep_loss
                + w_tv * tv_loss
            )

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'img': f"{img_loss.item():.4f}",
                'edge': f"{edge_loss.item():.4f}",
                'ssim': f"{ssim_loss.item():.4f}",
                'tv': f"{tv_loss.item():.4f}" + (f", perc={percep_loss.item():.4f}" if perceptual_net else "")
            })

        # Validation
        model.eval()
        psnr_vals, ssim_vals = [], []
        with torch.no_grad():
            for inp, gt in val_loader:
                inp, gt = inp.to(device), gt.to(device)
                pred, _ = model(inp)
                pred = torch.clamp(pred, 0, 1)
                for b in range(pred.size(0)):
                    pr = pred[b].cpu().permute(1,2,0).numpy()
                    gr = gt[b].cpu().permute(1,2,0).numpy()
                    mse = ((pr - gr) ** 2).mean()
                    psnr_vals.append(10 * np.log10(1.0 / mse) if mse > 0 else float('inf'))
                ssim_vals.append(ssim(pred, gt, data_range=1.0, size_average=True).item())
        avg_psnr = np.mean(psnr_vals)
        avg_ssim = np.mean(ssim_vals)
        print(f"[Epoch {epoch}] Val PSNR: {avg_psnr:.2f} dB, SSIM: {avg_ssim:.4f}")

        # Scheduler step
        scheduler.step(avg_ssim)

        # Early stopping
        if avg_ssim > best_ssim + config.es_delta:
            best_ssim = avg_ssim
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= config.es_patience:
            print(f"Early stopping at epoch {epoch}")
            break

        # Checkpoint
        if epoch % config.snapshot_freq == 0:
            os.makedirs(config.snapshots_folder, exist_ok=True)
            ckpt_path = os.path.join(
                config.snapshots_folder,
                f"epoch{epoch}_psnr{avg_psnr:.2f}.ckpt"
            )
            torch.save(model.state_dict(), ckpt_path)
            print("Saved:", ckpt_path)

    print("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # paths
    parser.add_argument('--input_images_path', type=str, default='./dataset/train/input/')
    parser.add_argument('--label_images_path', type=str, default='./dataset/train/label/')
    parser.add_argument('--val_input_path',   type=str, default='./dataset/val/input/')
    parser.add_argument('--val_label_path',   type=str, default='./dataset/val/label/')
    # hyperparams
    parser.add_argument('--lr',            type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--num_epochs',    type=int,   default=500)
    parser.add_argument('--loss_type',     type=str,   default='MSE', choices=['MSE','L1'])
    parser.add_argument('--edge_weight',   type=float, default=3.0)
    parser.add_argument('--ssim_weight',   type=float, default=0.5)
    parser.add_argument('--percep_weight', type=float, default=0.05)
    parser.add_argument('--percep_layer',  type=int,   default=8)
    parser.add_argument('--tv_weight',     type=float, default=0.01)
    parser.add_argument('--jitter',        type=float, default=0.1)
    parser.add_argument('--noise_std',     type=float, default=0.01)
    parser.add_argument('--grad_clip',     type=float, default=3.0)
    # scheduler
    parser.add_argument('--lr_factor',   type=float, default=0.5)
    parser.add_argument('--lr_patience', type=int,   default=10)
    # early stopping
    parser.add_argument('--es_patience', type=int,   default=20)
    parser.add_argument('--es_delta',    type=float, default=0.001)
    # train params
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--batch_size',   type=int,   default=8)
    parser.add_argument('--resize',       type=int,   default=256)
    parser.add_argument('--cuda_id',      type=int,   default=0)
    parser.add_argument('--snapshot_freq',type=int,   default=50)
    parser.add_argument('--snapshots_folder', type=str, default='./snapshots/')
    parser.add_argument('--num_workers',  type=int,   default=8)
    parser.add_argument('--smoke_test',   action='store_true')
    parser.add_argument('--smoke_size',   type=int,   default=500)

    config = parser.parse_args()
    train(config)

