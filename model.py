# import torch
# import torch.nn as nn
#
#
# class Edge_Detector(nn.Module):
#     def __init__(self):
#         super(Edge_Detector,self).__init__()
#         #self.kernel=torch.tensor([[[1.,1.,1.],[1.,-8.,1.],[1.,1.,1.]],[[1.,1.,1.],[1.,-8.,1.],[1.,1.,1.]],[[1.,1.,1.],[1.,-8.,1.],[1.,1.,1.]]])
#         self.conv1=nn.Conv2d(in_channels=3,out_channels=1,kernel_size=3,stride=1,padding=0,bias=False)
#         nn.init.constant_(self.conv1.weight,1)
#         nn.init.constant_(self.conv1.weight[0,0,1,1],-8)
#         nn.init.constant_(self.conv1.weight[0,1,1,1],-8)
#         nn.init.constant_(self.conv1.weight[0,2,1,1],-8)
#
#     def forward(self,x1):
#         edge_map=self.conv1(x1)
#         return edge_map
#
#
# class Res_Block(nn.Module):
#     def __init__(self):
#         super(Res_Block,self).__init__()
#         self.conv=nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1,bias=False)
#         self.relu=nn.ReLU(inplace=True)
#         self.bn=nn.BatchNorm2d(64)
#     def forward(self,x):
#
#         return torch.add(self.bn(self.conv(self.relu(self.bn(self.conv(x))))),x)
#
# class UResNet_P(nn.Module):
#     def __init__(self):
#         super(UResNet_P,self).__init__()
#         self.edge_detector=Edge_Detector()
#         self.residual_layer=self.stack_layer(Res_Block,16)
#         self.input=nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,stride=1,padding=1,bias=False)
#         self.output=nn.Conv2d(in_channels=64,out_channels=3,kernel_size=3,stride=1,padding=1,bias=False)
#         self.relu=nn.ReLU(inplace=True)
# #         for m in self.modules():
# #             if isinstance(m,nn.Conv2d):
# #                 n=m.kernel_size[0]*m.kernel_size[1]*m.out_channels
# #                 m.weight.data.norm(0,sqrt(2./n))
#     def stack_layer(self,block,num_of_layers):
#         layers=[]
#         for _ in range(num_of_layers):
#             layers.append(block())
#         return nn.Sequential(*layers)
#     def forward(self,x):
#         x=self.relu(self.input(x))
#         out=self.residual_layer(x)
#         out=torch.add(out,x)
#         out=self.output(out)
#         out = torch.sigmoid(out)
#         edge_map=self.edge_detector(out)
#         #out=torch.add(out,residual)
#         return out,edge_map


### model.py

import torch
import torch.nn as nn

class Edge_Detector(nn.Module):
    def __init__(self):
        super(Edge_Detector, self).__init__()
        # 用 padding=1 保持尺寸一致
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        # 初始化为拉普拉斯核
        nn.init.constant_(self.conv1.weight, 1)
        # 中心位置设为 -8
        with torch.no_grad():
            self.conv1.weight[:, :, 1, 1].fill_(-8)

    def forward(self, x):
        edge_map = self.conv1(x)
        return edge_map


class Res_Block(nn.Module):
    def __init__(self):
        super(Res_Block, self).__init__()
        self.conv = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(64)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv(x)))
        out = self.bn2(self.conv(out))
        return x + out


class UResNet_P(nn.Module):
    def __init__(self):
        super(UResNet_P, self).__init__()
        self.edge_detector = Edge_Detector()
        self.stack = self._make_layer(Res_Block, 16)
        self.input_conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.output_conv = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def _make_layer(self, block, n):
        layers = []
        for _ in range(n):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        # 保存原图做像素级跳连
        x_input = x
        # 特征域残差学习
        feat = self.relu(self.input_conv(x_input))
        res_feat = self.stack(feat)
        feat_sum = feat + res_feat
        # raw_out 是残差映射
        raw_out = self.output_conv(feat_sum)
        res = torch.sigmoid(raw_out)
        # 像素级残差加回
        out = torch.clamp(x_input + res, 0, 1)
        # 多尺度边缘检测由训练脚本处理
        edge_map = self.edge_detector(out)
        return out, edge_map
