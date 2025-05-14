# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.optim import lr_scheduler
# import numpy as np
# import torchvision
# from torchvision import transforms
# # import matplotlib.pyplot as plt
# import time
# import os
# import copy
# from model import Edge_Detector, Res_Block, UResNet_P
# import argparse
# from dataloader import TrainDataSet
#
# def train(config):
#     weight = 1.0
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     print("Using device:", device)
#     model = UResNet_P().to(device)
#     for param in model.edge_detector.parameters():
#         param.requires_grad = False
#     edge_detector = Edge_Detector().to(device)
#     for param in edge_detector.parameters():
#         param.requires_grad = False
#
#     if config.loss_type == 'MSE':
#         criterion = nn.MSELoss().to(device)
#     if config.loss_type == 'L1':
#         criterion == nn.L1Loss().to(device)
#
#     optimizer = optim.Adam(model.parameters(), lr=config.lr)
#     if config.train_mode == 'N' or 'P-S':
#     	scheduler = lr_scheduler.StepLR(optimizer,step_size=config.step_size,gamma=config.decay_rate)
#     if config.train_mode == 'P-A':
#     	scheduler = lr_scheduler.StepLR(optimizer,step_size=2*config.step_size,gamma=config.decay_rate)
#
#     transform_list = [transforms.Resize((config.resize, config.resize)), transforms.ToTensor()]
#     tsfms = transforms.Compose(transform_list)
#     train_dataset = TrainDataSet(config.input_images_path, config.label_images_path, tsfms)
#     train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False)
#     img_loss_lst = []
#     edge_loss_lst = []
#     total_loss_lst = []
#     for epoch in range(config.num_epochs):
#         img_loss_tmp = []
#         edge_loss_tmp = []
#         total_loss_tmp = []
#         if epoch > 1 and epoch % config.step_size == 0:
#             for param_group in optimizer.param_groups:
#                 param_group['lr'] = param_group['lr'] * 0.7
#
#         for input_img, label_img in train_dataloader:
#             input_img = input_img.to(device)
#             label_img = label_img.to(device)
#
#             if config.train_mode == 'N':
#                 model.zero_grad()
#                 generate_img, edge_map = model(input_img)
#                 loss = criterion(generate_img, label_img)
#                 img_loss_tmp.append(loss.item())
#                 loss.backward()
#                 optimizer.step()
#
#             if config.train_mode == 'P-A':
#
#                 for flag in range(2):
#                     model.zero_grad()
#                     generate_img, edge_map = model(input_img)
#                     if flag == 0:
#                         edge_label = edge_detector(label_img)
#                         edge_loss = criterion(edge_map, edge_label)
#                         edge_loss.backward()
#                     if flag == 1:
#                         img_loss = criterion(generate_img, label_img)
#                         img_loss.backward()
#
#                     scheduler.step()
#             if config.train_mode == 'P-S':
#                 model.zero_grad()
#                 generate_img, edge_map = model(input_img)
#                 img_loss = criterion(generate_img, label_img)
#                 edge_label = edge_detector(label_img)
#                 edge_loss = criterion(edge_map, edge_label)
#                 loss = img_loss + weight * edge_label
#                 total_loss_tmp.append(loss.item())
#                 loss.backward()
#
#         if config.train_mode == 'N':
#             img_loss_lst.append(np.mean(img_loss_tmp))
#
#         if config.train_mode == 'P-S':
#             total_loss_lst.append(np.mean(total_loss_tmp))
#
#         if config.train_mode == 'P-A':
#             img_loss_lst.append(np.mean(img_loss_tmp))
#             edge_loss_lst.append(np.mean(edge_loss_tmp))
#
#         if epoch % config.print_freq == 0:
#             if config.train_mode == 'N':
#                 print('epoch:[{}]/[{}], image loss:{}'.format(epoch, config.num_epochs, str(img_loss_lst[epoch])))
#             if config.train_mode == 'P-A':
#                 print('epoch:[{}]/[{}], image loss:{},edge difference loss:{}'.format(epoch, config.num_epochs,
#                                                                                       str(img_loss_lst[epoch]),
#                                                                                       str(edge_loss_lst[epoch])))
#             if config.train_mode == 'P-S':
#                 print('epoch:[{}]/[{}], total loss:{}'.format(epoch, config.num_epochs, str(total_loss_lst[epoch])))
#         if not os.path.exists(config.snapshots_folder):
#             os.mkdir(config.snapshots_folder)
#
#         if epoch % config.snapshot_freq == 0:
#             torch.save(model, config.snapshots_folder + 'model_epoch_{}.ckpt'.format(epoch))
#
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#
#     # Input Parameters
#     # parser.add_argument('--input_images_path', type=str, default="./data/input/",
#     #                     help='path of input images(underwater images) default:./data/input/')
#     # parser.add_argument('--label_images_path', type=str, default="./data/label/",
#     #                     help='path of label images(clear images) default:./data/label/')
#     # parser.add_argument('--input_images_path', type=str,
#     #                     default=r"D:\86178\Documents\zqtx\pythonProject5\backup\dataset\train\input",
#     #                     help='path of input images (underwater images)')
#     #
#     # parser.add_argument('--label_images_path', type=str,
#     #                     default=r"D:\86178\Documents\zqtx\pythonProject5\backup\dataset\train\label",
#     #                     help='path of label images (clear images)')
#
#
#     parser.add_argument('--input_images_path', type=str,
#                         default="D:/86178/Documents/zqtx/pythonProject5/backup/dataset/train/input/",
#                         help='path of input images (underwater images)')
#
#     parser.add_argument('--label_images_path', type=str,
#                         default="D:/86178/Documents/zqtx/pythonProject5/backup/dataset/train/label/",
#                         help='path of label images (clear images)')
#
#     parser.add_argument('--lr', type=float, default=0.0002)
#     parser.add_argument('--decay_rate', type=float, default=0.7, help='Learning rate decay default: 0.7')
#     parser.add_argument('--step_size', type=int, default=400, help="Period of learning rate decay")
#     parser.add_argument('--loss_type', type=str, default="MSE", help="loss type to train model, L1 or MSE default: MSE")
#     parser.add_argument('--num_epochs', type=int, default=2000)
#     parser.add_argument('--train_mode', type=str, default="P-A",
#                         help="N for UResNet;P-S for URseNet-P-S;P-A for UResnet-P-S. default:N")
#     parser.add_argument('--batch_size', type=int, default=1, help="default : 1")
#     parser.add_argument('--resize', type=int, default=256, help="resize images, default:resize images to 256*256")
#     parser.add_argument('--cuda_id', type=int, default=0, help="id of cuda device,default:0")
#     parser.add_argument('--print_freq', type=int, default=1)
#     parser.add_argument('--snapshot_freq', type=int, default=50)
#     parser.add_argument('--snapshots_folder', type=str, default="./snapshots/")
#     # parser.add_argument('--sample_output_folder', type=str, default="samples/")
#
#     config = parser.parse_args()
#
#     # if not os.path.exists(config.snapshots_folder):
#     #     os.mkdir(config.snapshots_folder)
#     # if not os.path.exists(config.sample_output_folder):
#     #     os.mkdir(config.sample_output_folder)
#
#     train(config)
#
#
#
#
#
#


import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import transforms
import os
from model import Edge_Detector, UResNet_P
import argparse
from dataloader import TrainDataSet
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# Enable cuDNN auto-tuner
torch.backends.cudnn.benchmark = True

def train(config):
    print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
    weight = config.edge_weight
    device = torch.device(f"cuda:{config.cuda_id}" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Model and edge detector
    model = UResNet_P().to(device)
    for param in model.edge_detector.parameters():
        param.requires_grad = False
    edge_detector = Edge_Detector().to(device)
    for param in edge_detector.parameters():
        param.requires_grad = False

    # Loss function
    if config.loss_type == 'MSE':
        criterion = nn.MSELoss().to(device)
    elif config.loss_type == 'L1':
        criterion = nn.L1Loss().to(device)
    else:
        raise ValueError("Unsupported loss type")

    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    if config.train_mode in ['N', 'P-S']:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.decay_rate)
    elif config.train_mode == 'P-A':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=2 * config.step_size, gamma=config.decay_rate)
    else:
        raise ValueError("Unsupported train mode")

    # Data transforms and dataset
    transform_list = [transforms.Resize((config.resize, config.resize)), transforms.ToTensor()]
    tsfms = transforms.Compose(transform_list)
    train_dataset = TrainDataSet(config.input_images_path, config.label_images_path, tsfms)

    # Optional smoke test subset
    if config.smoke_test:
        subset_size = min(config.smoke_size, len(train_dataset))
        train_dataset = Subset(train_dataset, list(range(subset_size)))
        print(f"Smoke test mode: using first {subset_size} samples")

    # DataLoader with parallel loading
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    # Training loop
    for epoch in range(config.num_epochs):
        pbar = tqdm(train_loader, desc=f"Epoch[{epoch}/{config.num_epochs}]")
        for input_img, label_img in pbar:
            input_img, label_img = input_img.to(device), label_img.to(device)

            optimizer.zero_grad()
            generate_img, edge_map = model(input_img)
            # Compute losses
            img_loss = criterion(generate_img, label_img)
            edge_target = edge_detector(label_img)
            edge_loss = criterion(edge_map, edge_target)
            if config.train_mode == 'P-S':
                loss = img_loss + weight * edge_loss
            elif config.train_mode == 'P-A':
                # 一次性计算并合并边缘 & 像素损失，然后一次 backward
                edge_target = edge_detector(label_img)
                edge_loss = criterion(edge_map, edge_target)
                img_loss = criterion(generate_img, label_img)
                loss = img_loss + edge_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
            else:
                loss = img_loss
                loss.backward()
                optimizer.step()
                scheduler.step()

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # Save checkpoint
        if (epoch + 1) % config.snapshot_freq == 0:
            os.makedirs(config.snapshots_folder, exist_ok=True)
            path = os.path.join(config.snapshots_folder, f"model_epoch_{epoch+1}.ckpt")
            torch.save(model.state_dict(), path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_images_path', type=str,
                        default="D:/86178/Documents/zqtx/pythonProject5/backup/dataset/train/input/",
                        help='path of input images')
    parser.add_argument('--label_images_path', type=str,
                        default="D:/86178/Documents/zqtx/pythonProject5/backup/dataset/train/label/",
                        help='path of label images')
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--decay_rate', type=float, default=0.7)
    parser.add_argument('--step_size', type=int, default=400)
    parser.add_argument('--loss_type', type=str, default="MSE")
    parser.add_argument('--num_epochs', type=int, default=2000)
    parser.add_argument('--train_mode', type=str, default="P-A",
                        help='"N", "P-S" or "P-A"')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--resize', type=int, default=256)
    parser.add_argument('--cuda_id', type=int, default=0)
    parser.add_argument('--print_freq', type=int, default=1)
    parser.add_argument('--snapshot_freq', type=int, default=50)
    parser.add_argument('--snapshots_folder', type=str, default="./snapshots/")
    parser.add_argument('--edge_weight', type=float, default=1.0)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--smoke_test', action='store_true',
                        help='run quick smoke test on small subset')
    parser.add_argument('--smoke_size', type=int, default=500,
                        help='subset size in smoke test')
    config = parser.parse_args()
    train(config)

