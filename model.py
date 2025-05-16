#
# import torch
# import torch.nn as nn
#
# class Edge_Detector(nn.Module):
#     def __init__(self):
#         super(Edge_Detector, self).__init__()
#         # 用 padding=1 保持尺寸一致
#         self.conv1 = nn.Conv2d(
#             in_channels=3,
#             out_channels=1,
#             kernel_size=3,
#             stride=1,
#             padding=1,
#             bias=False
#         )
#         # 初始化为拉普拉斯核
#         nn.init.constant_(self.conv1.weight, 1)
#         # 中心位置设为 -8
#         with torch.no_grad():
#             self.conv1.weight[:, :, 1, 1].fill_(-8)
#
#     def forward(self, x):
#         edge_map = self.conv1(x)
#         return edge_map
#
#
# class Res_Block(nn.Module):
#     def __init__(self):
#         super(Res_Block, self).__init__()
#         self.conv = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.bn2 = nn.BatchNorm2d(64)
#
#     def forward(self, x):
#         out = self.relu(self.bn1(self.conv(x)))
#         out = self.bn2(self.conv(out))
#         return x + out
#
#
# class UResNet_P(nn.Module):
#     def __init__(self):
#         super(UResNet_P, self).__init__()
#         self.edge_detector = Edge_Detector()
#         self.stack = self._make_layer(Res_Block, 16)
#         self.input_conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
#         self.output_conv = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1, bias=False)
#         self.relu = nn.ReLU(inplace=True)
#
#     def _make_layer(self, block, n):
#         layers = []
#         for _ in range(n):
#             layers.append(block())
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         # 保存原图做像素级跳连
#         x_input = x
#         # 特征域残差学习
#         feat = self.relu(self.input_conv(x_input))
#         res_feat = self.stack(feat)
#         feat_sum = feat + res_feat
#         # raw_out 是残差映射
#         raw_out = self.output_conv(feat_sum)
#         res = torch.sigmoid(raw_out)
#         # 像素级残差加回
#         out = torch.clamp(x_input + res, 0, 1)
#         # 多尺度边缘检测由训练脚本处理
#         edge_map = self.edge_detector(out)
#         return out, edge_map


# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

# Edge Detector unchanged
class Edge_Detector(nn.Module):
    def __init__(self):
        super(Edge_Detector, self).__init__()
        self.conv1 = nn.Conv2d(3,1,3,1,1,bias=False)
        nn.init.constant_(self.conv1.weight,1)
        with torch.no_grad():
            self.conv1.weight[:,:,1,1].fill_(-8)
    def forward(self,x): return self.conv1(x)

# Conv block for UNet
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

# UNet with skip connections
class UResNet_UNet(nn.Module):
    def __init__(self, base_ch=64):
        super().__init__()
        # Encoder
        self.enc1 = ConvBlock(3, base_ch)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(base_ch, base_ch*2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(base_ch*2, base_ch*4)
        self.pool3 = nn.MaxPool2d(2)
        # Bottleneck
        self.bottleneck = ConvBlock(base_ch*4, base_ch*8)
        # Decoder
        self.up3 = nn.ConvTranspose2d(base_ch*8, base_ch*4, 2,2)
        self.dec3 = ConvBlock(base_ch*8, base_ch*4)
        self.up2 = nn.ConvTranspose2d(base_ch*4, base_ch*2, 2,2)
        self.dec2 = ConvBlock(base_ch*4, base_ch*2)
        self.up1 = nn.ConvTranspose2d(base_ch*2, base_ch, 2,2)
        self.dec1 = ConvBlock(base_ch*2, base_ch)
        # Final conv
        self.final = nn.Conv2d(base_ch, 3, 1)
        # Edge detector
        self.edge_detector = Edge_Detector()
    def forward(self, x):
        # encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        # bottleneck
        b = self.bottleneck(self.pool3(e3))
        # decoder
        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        # output and residual
        raw = self.final(d1)
        res = torch.sigmoid(raw)
        out = torch.clamp(x + res, 0,1)
        edge_map = self.edge_detector(out)
        return out, edge_map

# Alias for training script
UResNet_P = UResNet_UNet

