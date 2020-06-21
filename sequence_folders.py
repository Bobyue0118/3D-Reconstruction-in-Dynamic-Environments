import torch.utils.data as data
import torch
import numpy as np
import quaternion
from scipy.misc import imread
from path import Path
import random
import os
import math
import matplotlib.pyplot as plt

from PIL import Image



def crawl_folders(root, folders_list, sequence_length):
    '''
    return a list which contains lots of samples : 
    sample = { 'rgb_tgt': rgb_imgs[i], 'rgb_ref_imgs': [], 'depth_tgt': depth_imgs[i], 'depth_ref_imgs': [], 'pose':[3,6]}
    '''
    sequence_set = []
    demi_length = (sequence_length-1)//2
    for folder in folders_list:
        folder = Path(root/folder)
        rgb_folder = folder/'rgb'
        depth_folder = folder/'depth'
        mask_folder = folder/'mask'
        rgb_imgs = sorted(rgb_folder.files('*.png'))
        depth_imgs = sorted(depth_folder.files('*.png'))
        mask_imgs = sorted(mask_folder.files('*.png'))

        pose_txt = np.loadtxt(folder/'pose.txt')

        if len(rgb_imgs) < sequence_length:
            continue

        for i in range(demi_length, len(rgb_imgs)-demi_length):

            sample = { 'rgb_tgt': rgb_imgs[i], 'rgb_ref_imgs': [], 'depth_tgt': depth_imgs[i], 'depth_ref_imgs': [], 'mask_tgt': mask_imgs[i], 'mask_ref_imgs': [], 'pose': []}
            for j in range(-demi_length, demi_length + 1):
                if j != 0:
                    sample['rgb_ref_imgs'].append(rgb_imgs[i+j])
                    sample['depth_ref_imgs'].append(depth_imgs[i+j])
                    sample['mask_ref_imgs'].append(mask_imgs[i+j])
                    if j < 0:
                        sample['pose'].append(torch.tensor(np.linalg.inv(generate_pose(pose_txt[0]))[:3,:]))
                    elif j > 0:
                        sample['pose'].append(torch.tensor(generate_pose(pose_txt[0])[:3,:]))
            sequence_set.append(sample)
    random.shuffle(sequence_set)
    return sequence_set # 

def quat2euler(array):
    x,y,z,w = array[0],array[1],array[2],array[3]

    r = math.atan2(2*(w*x+y*z),1-2*(x*x+y*y))
    tmp = 2*(w*y-z*x)
    if tmp >1:
        tmp = 1
    elif tmp <-1:
        tmp = -1
    p = math.asin(tmp)
    y = math.atan2(2*(w*z+x*y),1-2*(z*z+y*y))

    return r,p,y

def generate_pose(array):
    initial_pose = np.matrix(array)

    T_ros = np.matrix('-1 0 0 0;\
                        0 0 1 0;\
                        0 1 0 0;\
                        0 0 0 1')

    T_m = np.matrix('1.0157    0.1828   -0.2389    0.0113;\
                    0.0009   -0.8431   -0.6413   -0.0098;\
                    -0.3009    0.6147   -0.8085    0.0111;\
                        0         0         0    1.0000')

    t = initial_pose[0,0:3].transpose()
    q = np.quaternion(initial_pose[0,6],initial_pose[0,3],initial_pose[0,4],initial_pose[0,5])

    R = quaternion.as_rotation_matrix(q)

    T_0 = np.block([[R,t],[0, 0, 0, 1]])

    T_g = T_ros * T_0 * T_ros * T_m
    return T_g

def load_as_float(path):
    return imread(path).astype(np.float32)


class SequenceFolder(data.Dataset):
    """A sequence data loader where the files are arranged in this way:

        transform functions must take in a list a images and a numpy array (usually intrinsics matrix)
    """

    def __init__(self, root, intrinsics, seed=None, train=True, sequence_length=3, transform=None, target_transform=None):
        np.random.seed(seed)
        random.seed(seed)
        self.sequence_length = sequence_length
        self.root = Path(root)
        self.scenes = os.listdir(self.root)
        self.samples = crawl_folders(self.root, self.scenes, sequence_length)
        self.transform = transform
        self.intrinsics = intrinsics

    def __getitem__(self, index):
        sample = self.samples[index]
        # print(sample['rgb_tgt'])
        rgb_tgt_img = load_as_float(sample['rgb_tgt'])
        rgb_ref_imgs = [load_as_float(ref_img) for ref_img in sample['rgb_ref_imgs']]
        depth_tgt_img = load_as_float(sample['depth_tgt'])# [:, :, np.newaxis]
        depth_ref_imgs = [load_as_float(ref_img)for ref_img in sample['depth_ref_imgs']]
        mask_tgt_img = load_as_float(sample['mask_tgt'])# [:, :, np.newaxis]
        mask_ref_imgs = [load_as_float(ref_img)for ref_img in sample['mask_ref_imgs']]
        pose_list = sample['pose']
        '''
        plt.imshow(Image.open(sample['rgb_tgt']))
        plt.show()
        plt.imshow(Image.open(sample['depth_tgt']))
        plt.show()
        input()
        '''
        mask_tgt_img = torch.from_numpy(mask_tgt_img).float()
        mask_ref_imgs = [torch.from_numpy(img).float() for img in mask_ref_imgs]
        
        if self.transform is not None:
            imgs, intrinsics = self.transform([rgb_tgt_img] + rgb_ref_imgs + [depth_tgt_img] + depth_ref_imgs , np.copy(self.intrinsics)) # +[mask_tgt_img] + mask_ref_imgs
            rgb_tgt_img = imgs[0]
            rgb_ref_imgs = imgs[1:self.sequence_length]
            depth_tgt_img = imgs[self.sequence_length]
            depth_ref_imgs = imgs[self.sequence_length+1:2*self.sequence_length]
        

            # print(len(imgs),len(rgb_ref_imgs),len(depth_ref_imgs),)
        else:
            intrinsics = np.copy(self.intrinsics)
            '''
            plt.imshow((rgb_tgt_img/255))
            plt.show()
            plt.imshow((depth_tgt_img))
            plt.show()
            input()
            '''
            rgb_tgt_img = ArrayToTensor(rgb_tgt_img)
            rgb_ref_imgs = [ArrayToTensor(ref_img) for ref_img in rgb_ref_imgs]
            depth_tgt_img = ArrayToTensor(depth_tgt_img)
            depth_ref_imgs = [ArrayToTensor(ref_img) for ref_img in depth_ref_imgs]


        
        return rgb_tgt_img, rgb_ref_imgs, depth_tgt_img, depth_ref_imgs,mask_tgt_img,mask_ref_imgs, intrinsics, np.linalg.inv(intrinsics), pose_list

    def __len__(self):
        return len(self.samples)


def ArrayToTensor(im):


    if im.shape[-1] ==3 :
        im = np.transpose(im, (2, 0, 1))
    # handle numpy array
    tensors = (torch.from_numpy(im).float()/255)
    # print(4, len(tensors))

    return tensors

class ValSequenceFolder(data.Dataset):
    """A sequence data loader where the files are arranged in this way:

        transform functions must take in a list a images and a numpy array (usually intrinsics matrix)
    """

    def __init__(self, root, intrinsics, seed=None, train=False, sequence_length=3, transform=None, target_transform=None):
        np.random.seed(seed)
        random.seed(seed)
        self.sequence_length = sequence_length
        self.root = Path(root)
        self.scenes = [self.root/'rgbd_bonn_balloon']
        self.samples = crawl_folders(self.root, self.scenes, sequence_length)
        self.transform = transform
        self.intrinsics = intrinsics

    def __getitem__(self, index):
        sample = self.samples[index]
        # print(sample['rgb_tgt'])
        rgb_tgt_img = load_as_float(sample['rgb_tgt'])
        rgb_ref_imgs = [load_as_float(ref_img) for ref_img in sample['rgb_ref_imgs']]
        depth_tgt_img = load_as_float(sample['depth_tgt'])# [:, :, np.newaxis]
        depth_ref_imgs = [load_as_float(ref_img)for ref_img in sample['depth_ref_imgs']]
        mask_tgt_img = load_as_float(sample['mask_tgt'])# [:, :, np.newaxis]
        mask_ref_imgs = [load_as_float(ref_img)for ref_img in sample['mask_ref_imgs']]
        pose_list = sample['pose']
        '''
        plt.imshow(Image.open(sample['rgb_tgt']))
        plt.show()
        plt.imshow(Image.open(sample['depth_tgt']))
        plt.show()
        input()
        '''
        mask_tgt_img = torch.from_numpy(mask_tgt_img).float()
        mask_ref_imgs = [torch.from_numpy(img).float() for img in mask_ref_imgs]
        
        if self.transform is not None:
            imgs, intrinsics = self.transform([rgb_tgt_img] + rgb_ref_imgs + [depth_tgt_img] + depth_ref_imgs , np.copy(self.intrinsics)) # +[mask_tgt_img] + mask_ref_imgs
            rgb_tgt_img = imgs[0]
            rgb_ref_imgs = imgs[1:self.sequence_length]
            depth_tgt_img = imgs[self.sequence_length]
            depth_ref_imgs = imgs[self.sequence_length+1:2*self.sequence_length]
        

            # print(len(imgs),len(rgb_ref_imgs),len(depth_ref_imgs),)
        else:
            intrinsics = np.copy(self.intrinsics)
            '''
            plt.imshow((rgb_tgt_img/255))
            plt.show()
            plt.imshow((depth_tgt_img))
            plt.show()
            input()
            '''
            rgb_tgt_img = ArrayToTensor(rgb_tgt_img)
            rgb_ref_imgs = [ArrayToTensor(ref_img) for ref_img in rgb_ref_imgs]
            depth_tgt_img = ArrayToTensor(depth_tgt_img)
            depth_ref_imgs = [ArrayToTensor(ref_img) for ref_img in depth_ref_imgs]


        
        return rgb_tgt_img, rgb_ref_imgs, depth_tgt_img, depth_ref_imgs,mask_tgt_img,mask_ref_imgs, intrinsics, np.linalg.inv(intrinsics), pose_list

    def __len__(self):
        return len(self.samples)
