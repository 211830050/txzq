# app.py
# -*- coding: utf-8 -*-

import os
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinterdnd2 import DND_FILES, TkinterDnD
from PIL import Image, ImageTk

import torch
from torchvision import transforms

from model import UNetPlus  # 请保证 sys.path 中能找到你的 model.py

# ———— 配置区 ————
CKPT_PATH     = './snapshots/model60.pth'  # 训练好的权重文件
DEVICE         = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
RESIZE         = 256
BASE_CHANNELS  = 64
# ——————————

class UnderwaterEnhancerApp(TkinterDnD.Tk):
    def __init__(self):
        super().__init__()
        self.title('水下图像增强 Demo')
        self.geometry('800x400')

        # 1. 加载模型
        self.model = UNetPlus(base_channels=BASE_CHANNELS).to(DEVICE)
        self.model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
        self.model.eval()

        # 2. 定义预处理
        self.preprocess = transforms.Compose([
            transforms.Resize((RESIZE, RESIZE)),
            transforms.ToTensor()
        ])

        # 3. 构建界面
        self._build_ui()

        # 占位
        self.input_img = None
        self.input_path = None

    def _build_ui(self):
        # 左右两侧 Frame
        left_frame  = tk.Frame(self, width=400, height=400)
        right_frame = tk.Frame(self, width=400, height=400)
        left_frame.pack(side='left',  fill='both', expand=True)
        right_frame.pack(side='right', fill='both', expand=True)

        # 左侧：拖拽 / 点击 + 按钮
        self.drop_area = tk.Label(
            left_frame,
            text='拖拽图片到此处\n或点击此处选择',
            bg='lightgray',
            width=40, height=20
        )
        self.drop_area.pack(padx=10, pady=10, fill='both', expand=True)

        # 注册拖放
        self.drop_area.drop_target_register(DND_FILES)
        self.drop_area.dnd_bind('<<Drop>>', self._on_drop)
        self.drop_area.bind('<Button-1>', self._on_click)

        self.enhance_btn = tk.Button(
            left_frame, text='增强图像', command=self._on_enhance
        )
        self.enhance_btn.pack(pady=5)

        # 右侧：显示增强结果
        self.result_area = tk.Label(
            right_frame,
            text='增强结果会显示在这里',
            bg='white',
            width=40, height=20
        )
        self.result_area.pack(padx=10, pady=10, fill='both', expand=True)

    def _on_click(self, event):
        """点击弹出文件选择对话框"""
        path = filedialog.askopenfilename(
            filetypes=[('Image', '*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff')]
        )
        if path:
            self._load_input(path)

    def _on_drop(self, event):
        """拖拽回调，event.data 格式如 '{C:/path/to/img.png}'"""
        path = event.data
        # Windows 下会带大括号和空格分隔多文件，这里只取第一个
        if path.startswith('{') and path.endswith('}'):
            path = path[1:-1]
        path = path.strip().split()[0]
        self._load_input(path)

    def _load_input(self, path):
        """加载并在左侧显示原图"""
        try:
            img = Image.open(path).convert('RGB')
        except Exception as e:
            messagebox.showerror('错误', f'无法打开文件：{e}')
            return
        img = img.resize((RESIZE, RESIZE), Image.BILINEAR)
        self.input_img  = img
        self.input_path = path

        tkimg = ImageTk.PhotoImage(img)
        self.drop_area.configure(image=tkimg, text='')
        # 必须保持引用，否则图片不显示
        self.drop_area.image = tkimg

    def _on_enhance(self):
        """调用模型增强并在右侧显示结果"""
        if self.input_img is None:
            messagebox.showwarning('提示', '请先加载一张图片')
            return

        tensor = self.preprocess(self.input_img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = self.model(tensor)           # [1,3,H,W]
            out = torch.clamp(out, 0.0, 1.0)
        # 转 PIL
        out_np = out[0].cpu().permute(1, 2, 0).numpy()
        out_img = Image.fromarray((out_np * 255).astype('uint8'))

        tkout = ImageTk.PhotoImage(out_img.resize((RESIZE, RESIZE), Image.BILINEAR))
        self.result_area.configure(image=tkout, text='')
        self.result_area.image = tkout

if __name__ == '__main__':
    # Windows 下需要这一行支持多进程拖放
    import multiprocessing
    multiprocessing.freeze_support()

    # 请先安装依赖：pip install tkinterdnd2 pillow torch torchvision
    app = UnderwaterEnhancerApp()
    app.mainloop()
