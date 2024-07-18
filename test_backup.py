import cv2
import os
import torch.utils.data
import numpy as np
from models.common import post_process_output
from utils.dataset_processing import evaluation
import streamlit as st

device = torch.device("cuda:0")

modelpath = "/home/ddsoda/YkkPj/ggcnn-master/output/models/240712_1756_/epoch_41_iou_0.80"

load_no = 1
formatted_no = str(load_no).zfill(4)
depth_path = f"/home/ddsoda/YkkPj/RealSense/save_file/pcd{formatted_no}d.png"
rgb_path = f"/home/ddsoda/YkkPj/RealSense/save_file/pcd{formatted_no}r.png"

def crop_image(datapath):
    # 打开图片
    img = cv2.imread(datapath,cv2.IMREAD_UNCHANGED)
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
        # 如果在处理过程中发生异常，打印异常信息
        print(e)

if __name__ == '__main__':
    # 加载训练完的模型
    net = torch.load(modelpath)
    net.to(device)
    # 判断文件存在
    if os.path.exists(depth_path):
        depth_img = crop_image(depth_path)
        depth_img = depth_img / 1000
        depth_new = interp_data(depth_img)
        depth_new = np.clip((depth_new - depth_new.mean()), -1, 1)
        depth_input = torch.from_numpy(depth_new).unsqueeze(0).unsqueeze(1).float().to(device)
        rgb_img = crop_image(rgb_path)
        rgb_input = torch.from_numpy(rgb_img)
        with torch.no_grad():
            pos_pred, cos_pred, sin_pred, width_pred = net(depth_input)
            # 后处理网络输出，得到Q图像、角度图像和宽度图像
            q_img, ang_img, width_img = post_process_output(pos_pred, cos_pred, sin_pred, width_pred)
            depth_visible = depth_input.squeeze().to("cpu")
            # test_rgb : 测试图像rgb
            # test_depth : 测试深度图
            # q_img : 预测的Q图像
            # ang_img : 预测的角度图像
            # no_grasps : 预测的抓取数量
            # grasp_width_img : 预测的宽度图像
            evaluation.plot_output(rgb_img, depth_visible, q_img, ang_img, no_grasps=1, grasp_width_img=width_img)

        # st.image(rgb_img, caption=f'Image for pcd{formatted_no}r.png', use_column_width=True)
    else:
        st.error(f"File not found: {depth_path}")