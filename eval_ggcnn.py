import argparse
import logging

import torch.utils.data

from models.common import post_process_output
from utils.dataset_processing import evaluation, grasp
from utils.data import get_dataset

logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate GG-CNN')

    # Network
    parser.add_argument('--network', type=str, default='./output/models/240712_1408_/epoch_47_iou_0.74', help='Path to saved network to evaluate')

    # Dataset & Data & Training
    parser.add_argument('--dataset', type=str, default='cornell', help='Dataset Name ("cornell" or "jaquard")')
    parser.add_argument('--dataset-path', type=str, default='/home/ddsoda/YkkPj/00_dataset/cornell_all',help='Path to dataset')
    parser.add_argument('--use-depth', type=int, default=1, help='Use Depth image for evaluation (1/0)')
    parser.add_argument('--use-rgb', type=int, default=0, help='Use RGB image for evaluation (0/1)')
    parser.add_argument('--augment', action='store_true', help='Whether data augmentation should be applied')
    parser.add_argument('--split', type=float, default=0.9, help='Fraction of data for training (remainder is validation)')
    parser.add_argument('--ds-rotate', type=float, default=0.0,
                        help='Shift the start point of the dataset to use a different test/train split')
    parser.add_argument('--num-workers', type=int, default=8, help='Dataset workers')

    parser.add_argument('--n-grasps', type=int, default=1, help='Number of grasps to consider per image')
    parser.add_argument('--iou-eval', action='store_true', default=True ,help='Compute success based on IoU metric.')
    parser.add_argument('--jacquard-output', action='store_true', help='Jacquard-dataset style output')
    parser.add_argument('--vis', action='store_true', default=True, help='Visualise the network output')

    args = parser.parse_args()

    if args.jacquard_output and args.dataset != 'jacquard':
        raise ValueError('--jacquard-output can only be used with the --dataset jacquard option.')
    if args.jacquard_output and args.augment:
        raise ValueError('--jacquard-output can not be used with data augmentation.')

    return args

if __name__ == '__main__':
    # 解析命令行参数
    args = parse_args()

    # 加载神经网络模型
    net = torch.load(args.network)
    # 设置设备为GPU 0
    device = torch.device("cuda:0")

    # 加载数据集
    logging.info('Loading {} Dataset...'.format(args.dataset.title()))
    # 获取指定的数据集类
    Dataset = get_dataset(args.dataset)
    # 初始化测试数据集
    test_dataset = Dataset(args.dataset_path,
                           start=args.split,
                           end=1.0,
                           ds_rotate=args.ds_rotate,
                           random_rotate=args.augment,
                           random_zoom=args.augment,
                           include_depth=args.use_depth,
                           include_rgb=args.use_rgb)
    # 创建dataloader
    test_data = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=1,
                                            shuffle=False,
                                            num_workers=args.num_workers)
    logging.info('Done')

    # 初始化结果统计
    results = {'correct': 0, 'failed': 0}

    # 如果需要输出Jacquard格式的结果
    if args.jacquard_output:
        jo_fn = args.network + '_jacquard_output.txt'
        # 清空或创建输出文件
        with open(jo_fn, 'w') as f:
            pass

    # 开始预测过程，不计算梯度以节省内存和加速
    with torch.no_grad():
        # 遍历测试数据
        for idx, (x, y, didx, rot, zoom) in enumerate(test_data):
            # 输出当前处理进度
            logging.info('Processing {}/{}'.format(idx+1, len(test_data)))
            # 将输入数据移动到GPU上
            xc = x.to(device)
            # 将标签数据移动到GPU上
            yc = [yi.to(device) for yi in y]
            # 计算损失
            lossd = net.compute_loss(xc, yc)

            # 后处理网络输出，得到Q图像、角度图像和宽度图像
            q_img, ang_img, width_img = post_process_output(lossd['pred']['pos'],
                                                            lossd['pred']['cos'],
                                                            lossd['pred']['sin'],
                                                            lossd['pred']['width'])

            # 如果需要进行IoU评估
            if args.iou_eval:
                # 计算IoU匹配
                s = evaluation.calculate_iou_match(q_img, ang_img,
                                                   test_data.dataset.get_gtbb(didx, rot, zoom),
                                                   no_grasps=args.n_grasps,
                                                   grasp_width=width_img,
                                                   )
                # 更新结果统计
                if s:
                    results['correct'] += 1
                else:
                    results['failed'] += 1

            # 如果需要输出Jacquard格式的结果
            if args.jacquard_output:
                # 检测抓取点
                grasps = grasp.detect_grasps(q_img, ang_img,
                                             width_img=width_img,
                                             no_grasps=1)
                # 写入Jacquard格式的输出文件
                with open(jo_fn, 'a') as f:
                    for g in grasps:
                        f.write(test_data.dataset.get_jname(didx) + '\n')
                        f.write(g.to_jacquard(scale=1024 / 300) + '\n')

            # 如果需要可视化
            if args.vis:
                # 可视化输出
                evaluation.plot_output(test_data.dataset.get_rgb(didx, rot, zoom, normalise=False),
                                       test_data.dataset.get_depth(didx, rot, zoom),
                                       q_img, ang_img,
                                       no_grasps=args.n_grasps,
                                       grasp_width_img = width_img)

    # 如果进行了IoU评估，输出结果
    if args.iou_eval:
        logging.info('IOU Results: %d/%d = %f' % (results['correct'],
                                                  results['correct'] + results['failed'],
                                                  results['correct'] / (results['correct'] + results['failed'])))

    # 如果输出了Jacquard格式的结果，记录文件位置
    if args.jacquard_output:
        logging.info('Jacquard output saved to {}'.format(jo_fn))
