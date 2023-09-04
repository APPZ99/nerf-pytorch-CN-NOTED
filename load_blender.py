import os
import torch
import numpy as np
import imageio 
import json
import torch.nn.functional as F
import cv2


trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w


def load_blender_data(basedir, half_res=False, testskip=1):

    # 三类数据 训练集、验证集、测试集
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        # 依次读取三个数据集的内容
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        # testskip 作用是读取数据集过程中是否跳着读
        if s=='train' or testskip==0:
            skip = 1
        else:
            skip = testskip
            
        # frames 中记录图片的路径、旋转矩阵
        for frame in meta['frames'][::skip]:
            # 读取图片
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            imgs.append(imageio.imread(fname))
            # 读取图片旋转矩阵
            poses.append(np.array(frame['transform_matrix']))
        # RGB 为图片色彩通道，A 为 Alpha，指图片的色彩空间，通常用作不透明参数
        # TODO:将每一张图片归一化到 0~1.
        imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)
    
    # 结合counts 列表将所有图片分为 train、val、test 三个部分
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
    
    # 将所有图片按照 x 轴进行拼接
    # NOTE: np.concatente()：对数组按照 axis 进行拼接
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    
    H, W = imgs[0].shape[:2]
    # 读取相机角度以及计算焦距
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    
    # 一共渲染40帧，每9度一个间隔
    # NOTE: np.linspace()：生成等间隔数组
    # NOTE: torch.stack()：对torch进行拼接
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
    
    # 如果使用图片的一半大小
    if half_res:
        H = H//2
        W = W//2
        # 焦距对应也要缩小一半
        focal = focal/2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()

        
    return imgs, poses, render_poses, [H, W, focal], i_split


