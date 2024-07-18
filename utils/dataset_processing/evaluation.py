import numpy as np
import matplotlib.pyplot as plt
from .grasp import GraspRectangles, detect_grasps
import math
import streamlit as st

def plot_output(rgb_img, depth_img, grasp_q_img, grasp_angle_img, figure, no_grasps=1, grasp_width_img=None, contrast=False):
    '''
    绘制GG-CNN的输出
    ：参数rgb_img:rgb图像
    ：param depth_img：深度图像
    ：参数grasp_q_img:GG-CNN的q输出
    ：参数grasp_angle_img:GG-CNN的角度输出
    ：param no_charses：要绘制的最大抓取次数
    ：param grasp_width_img：（可选）GG-CNN的宽度输出
    '''

    gs = detect_grasps(grasp_q_img, grasp_angle_img, width_img=grasp_width_img, no_grasps=no_grasps)
    if not contrast:
        ax_1 = figure.add_subplot(1, 3, 1, label='ax_1')
    else:
        ax_2 = figure.add_subplot(1, 3, 2, label='ax_2')

    # 定义文本框的样式
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # 字号
    font_size = 10
    for g in gs:
        xdelta = np.cos(g.angle)
        ydelta = np.sin(g.angle)
        y1 = g.center[0] + g.length / 3 * ydelta
        x1 = g.center[1] - g.length / 3 * xdelta
        y2 = g.center[0] - g.length / 3 * ydelta
        x2 = g.center[1] + g.length / 3 * xdelta
        grasp_center = g.center
        grasp_length = g.length
        grasp_angle = g.angle
        # g.plot(ax)
    ax = ax_1 if not contrast else ax_2
    ax.imshow(rgb_img)
    ax.text(0.8, 0.8, f'Center: {grasp_center}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, bbox=props, fontsize=font_size)
    # 显示抓取长度
    ax.text(0.8, 0.7, f'Length: {(grasp_length*2/3):.2f} pixels', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, bbox=props, fontsize=font_size)
    # 显示抓取角度
    ax.text(0.8, 0.6, f'Angle: {(grasp_angle*180/math.pi):.2f}°', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, bbox=props, fontsize=font_size)
    ax.plot([x1, x2], [y1, y2], color='red')
    ax.scatter(grasp_center[1], grasp_center[0], color='blue', s=20)
    ax.set_title(f"GGCNN{'-1' if not contrast else '-2'} RGB")
    ax.axis('off')

    if contrast:
        ax_3 = figure.add_subplot(1, 3, 3, label='ax_3')
        ax_3.imshow(depth_img, cmap='gray')
        ax_3.set_title('Depth')
        ax_3.axis('off')

    '''
    ax = fig.add_subplot(2, 2, 3)
    plot = ax.imshow(grasp_q_img, cmap='jet', vmin=0, vmax=1)
    ax.set_title('Q')
    ax.axis('off')
    plt.colorbar(plot)

    ax = fig.add_subplot(2, 2, 4)
    plot = ax.imshow(grasp_angle_img, cmap='hsv', vmin=-np.pi / 2, vmax=np.pi / 2)
    ax.set_title('Angle')
    ax.axis('off')
    plt.colorbar(plot)
    '''
    # plt.show()

def calculate_iou_match(grasp_q, grasp_angle, ground_truth_bbs, no_grasps=1, grasp_width=None):
    """
    Calculate grasp success using the IoU (Jacquard) metric (e.g. in https://arxiv.org/abs/1301.3592)
    A success is counted if grasp rectangle has a 25% IoU with a ground truth, and is withing 30 degrees.
    :param grasp_q: Q outputs of GG-CNN (Nx300x300x3)
    :param grasp_angle: Angle outputs of GG-CNN
    :param ground_truth_bbs: Corresponding ground-truth BoundingBoxes
    :param no_grasps: Maximum number of grasps to consider per image.
    :param grasp_width: (optional) Width output from GG-CNN
    :return: success
    """

    if not isinstance(ground_truth_bbs, GraspRectangles):
        gt_bbs = GraspRectangles.load_from_array(ground_truth_bbs)
    else:
        gt_bbs = ground_truth_bbs
    gs = detect_grasps(grasp_q, grasp_angle, width_img=grasp_width, no_grasps=no_grasps)
    for g in gs:
        if g.max_iou(gt_bbs) > 0.25:
            return True
    else:
        return False