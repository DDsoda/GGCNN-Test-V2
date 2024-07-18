import cv2
import os
import math
import torch.utils.data
import numpy as np
from models.common import post_process_output
from utils.dataset_processing import evaluation
from utils.data.test_data import TestDataset
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from utils.dataset_processing.grasp import detect_grasps
import tkinter as tk
import streamlit as st

# 全局变量
figures = []

# 参数定义
device = torch.device("cuda:0")
modelpath = '/home/ddsoda/YkkPj/ggcnn-ykk/output/models/240712_1756_/epoch_41_iou_0.80'
modelpath_contrast = '/home/ddsoda/YkkPj/ggcnn-ykk/output/models/240712_1408_/epoch_47_iou_0.74'
data_path = '/home/ddsoda/YkkPj/RealSense/save_file'
preprocess_outdir = '/home/ddsoda/YkkPj/ggcnn-ykk/test_folder'
preprocess_need = False
comtrast_need = True

# 中心裁剪变换
def crop_image(img):
    # 获取图像的宽度和高度
    height, width = img.shape[:2]
    # 计算裁剪区域的左上角和右下角坐标
    left = (width - 300) // 2
    top = (height - 300) // 2
    right = left + 300
    bottom = top + 300
    # 裁剪图像
    cropped_img = img[top:bottom, left:right]
    # 显示裁剪后的图像
    # cv2.imshow('Cropped Image', cropped_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return cropped_img
def interp_data(depth_image):
    # 初始化插值计数器
    fn = 0
    # 使用输入的深度图像创建一个新的数据数组，用于存储处理后的结果
    new_data = depth_image
    try:
        # 遍历图像的每一个像素
        for i in range(new_data.shape[0]):
            for j in range(new_data.shape[1]):
                # 如果当前像素值为0，需要进行插值处理
                if new_data[i, j] == 0:
                    # 如果当前列是最左边，向右查找第一个非0值
                    if j == 0:
                        d = j
                        while new_data[i, d] == 0:
                            d += 1
                        new_data[i, j] = new_data[i, d]
                    # 如果当前列不是最左边，使用左侧第一个非0值进行插值
                    else:
                        new_data[i, j] = new_data[i, j - 1]
                    # 插值计数器加1
                    fn += 1
        # 返回处理后的深度图像数组
        return new_data
    except Exception as e:
        # 如果发生异常，打印异常信息
        print(e)

def preprocess(rgb_files, depth_files):
    if not os.path.exists(preprocess_outdir):
        os.makedirs(preprocess_outdir)
    for rgb_file, depth_file in zip(rgb_files, depth_files):
        # 加载图像
        rgb_img = cv2.imread(os.path.join(data_path, rgb_file), cv2.IMREAD_UNCHANGED)
        depth_img = cv2.imread(os.path.join(data_path, depth_file), cv2.IMREAD_UNCHANGED)
        depth_img = depth_img.astype(np.float32)
        # 中心裁剪
        cropped_depth_img = crop_image(depth_img)
        new_rgb_img = crop_image(rgb_img)
        # 深度图像单位转换
        new_depth_img = cropped_depth_img / 1000
        # 深度图像应用零像素校正
        new_depth_img = interp_data(new_depth_img)
        # 深度图像归一化
        new_depth_img = np.clip((new_depth_img - new_depth_img.mean()), -1, 1)
        # 保存图像
        cv2.imwrite(os.path.join(preprocess_outdir, rgb_file), new_rgb_img)
        # cv2.imwrite(os.path.join(preprocess_outdir, depth_file), new_depth_img)
        np.save(os.path.join(preprocess_outdir, depth_file.replace('.png', '.npy')), new_depth_img)

def run(contrast=False):
    if (preprocess_need and not contrast):
        rgb_original_files = sorted([p for p in os.listdir(data_path) if p.endswith('r.png')])
        depth_original_files = sorted([p for p in os.listdir(data_path) if p.endswith('d.png')])
        preprocess(rgb_original_files, depth_original_files)
    # 实例化Dataset和DataLoader
    testset = TestDataset(root_dir_data=preprocess_outdir)
    data_loader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)
    # 初始设定绘图列表
    if not contrast:
        for _ in range(len(testset)):
            fig = plt.figure(figsize=(20, 20), dpi=300)
            figures.append(fig)
    # 加载训练完的模型
    net = torch.load(modelpath) if not contrast else torch.load(modelpath_contrast)
    net.to(device)
    # 遍历DataLoader
    for batch_idx, (rgb,depth) in enumerate(data_loader):
        depth=depth.unsqueeze(1).to(device)
        print(f'Batch index: {batch_idx}')
        print(f'RGB shape: {rgb.shape}')
        print(f'Depth shape: {depth.shape}')
        with torch.no_grad():
            pos_pred, cos_pred, sin_pred, width_pred = net(depth)
            # 后处理网络输出，得到Q图像、角度图像和宽度图像
            q_img, ang_img, width_img = post_process_output(pos_pred, cos_pred, sin_pred, width_pred)
            depth_visible = depth.squeeze().to("cpu")
            # test_rgb : 测试图像rgb
            # test_depth : 测试深度图
            # q_img : 预测的Q图像
            # ang_img : 预测的角度图像
            # no_grasps : 预测的抓取数量
            # grasp_width_img : 预测的宽度图像
            for i in range(len(rgb)):
                fig_idx = batch_idx * data_loader.batch_size + i
                print(f'Figure index: {fig_idx}')
                evaluation.plot_output(rgb[i], depth_visible[i], q_img[i], ang_img[i], figure=figures[fig_idx], no_grasps=1, grasp_width_img=width_img[i], contrast=contrast)

if __name__ == '__main__':
    run()
    if comtrast_need:
        run(contrast=True)
    for fig in figures:
        st.pyplot(fig)