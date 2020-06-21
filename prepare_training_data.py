from __future__ import division
import argparse
import scipy.misc
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from path import Path
from PIL import Image
from PIL.Image import NEAREST, BILINEAR
import os
import re

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", metavar='DIR',
                    help='path to original dataset')
parser.add_argument("--with-gt", action='store_true',
                    help="If available (e.g. with KITTI), will store ground truth along with images, for validation")
parser.add_argument("--dump-root", type=str, required=False, help="Where to dump the data")
parser.add_argument("--height", type=int, default=384, help="image height") #from 480 to 384
parser.add_argument("--width", type=int, default=512, help="image width")  #from 640 to 512


args = parser.parse_args()

def main():
    folder_list = os.listdir(args.dataset_dir)

    for one_scene in tqdm(folder_list):

        #-----------associate-----------------------
        rgb_list = read_file_list(args.dataset_dir+'/'+one_scene+'/rgb.txt')
        depth_list = read_file_list(args.dataset_dir+'/'+one_scene+'/depth.txt')

        rgb_depth_matches = associate(rgb_list, depth_list,float(0),float(0.02))    
        with open(args.dataset_dir+'/'+one_scene+"/rgb_depth_matches.txt","w") as f:
            for a,b in rgb_depth_matches:
                f.write("%f %s %f %s"%(a," ".join(rgb_list[a]),b-float(0)," ".join(depth_list[b])) +'\n')
        
        gt_list = read_file_list(args.dataset_dir+'/'+one_scene+'/groundtruth.txt')
        rgb_depth_matches = read_file_list(args.dataset_dir+'/'+one_scene+'/rgb_depth_matches.txt')

        all_match = associate(rgb_depth_matches, gt_list, float(0),float(0.02))  
        with open(args.dataset_dir+'/'+one_scene+"/match_all.txt","w") as f:
            for a,b in all_match:
                f.write("%f %s %f %s"%(a," ".join(rgb_depth_matches[a]),b-float(0)," ".join(gt_list[b])) +'\n')
        #---------------------------------

        rgb_image_dir = Path(args.dump_root+'/'+one_scene+'/'+'rgb')
        rgb_image_dir.makedirs_p()
        depth_image_dir = Path(args.dump_root+'/'+one_scene+'/'+'depth')
        depth_image_dir.makedirs_p()

        # -----------load images --------------------
        rgb_images = os.listdir(args.dataset_dir+'/'+one_scene+'/rgb')
        rgb_images = sorted(rgb_images, key=embedded_numbers)
        depth_images = os.listdir(args.dataset_dir+'/'+one_scene+'/depth')
        depth_images = sorted(depth_images, key=embedded_numbers)

        with open(args.dataset_dir+'/'+one_scene+'/match_all.txt','r') as all_groups:
            all_lines = all_groups.readlines()
            with open ((args.dump_root+'/'+one_scene+'/'+'pose.txt'),'w') as pose_txt:
                scene_id = 0
                for line in all_lines:
                    tmp = line.split(' ')
                    rgb_id, depth_id, pose = tmp[1], tmp[3], tmp[5:]
                    k=' '.join([str(j) for j in pose])
                    pose_txt.write(k.lstrip('\n'))


                    rgb_image = resize_image_rgb(args.dataset_dir+'/'+one_scene+'/'+rgb_id) # numpy array
                    depth_image = resize_image_depth(args.dataset_dir+'/'+one_scene+'/'+depth_id)
                    Image.fromarray(rgb_image).convert("RGB").save(rgb_image_dir/str(scene_id).zfill(4)+'.png')
                    (depth_image).save(depth_image_dir/str(scene_id).zfill(4)+'.png')
                    scene_id += 1

              

def embedded_numbers(s):
    re_digits = re.compile(r'(\d+)')
    pieces = re_digits.split(s)           
    pieces[1::2] = map(int, pieces[1::2])    
    return pieces

def resize_image_depth(img_file):
    img = Image.open(img_file)
    img = img.resize ((args.width, args.height), resample=BILINEAR)
    return img

def resize_image_rgb(img_file):
    img = scipy.misc.imread(img_file)
    img = scipy.misc.imresize(img, (args.height, args.width))
    return img

def read_file_list(filename):
    """
    Reads a trajectory from a text file. 
    
    File format:
    The file format is "stamp d1 d2 d3 ...", where stamp denotes the time stamp (to be matched)
    and "d1 d2 d3.." is arbitary data (e.g., a 3D position and 3D orientation) associated to this timestamp. 
    
    Input:
    filename -- File name
    
    Output:
    dict -- dictionary of (stamp,data) tuples
    
    """
    file = open(filename)
    data = file.read()
    lines = data.replace(","," ").replace("\t"," ").split("\n") 
    list = [[v.strip() for v in line.split(" ") if v.strip()!=""] for line in lines if len(line)>0 and line[0]!="#"]
    list = [(float(l[0]),l[1:]) for l in list if len(l)>1]
    return dict(list)

def associate(first_list, second_list,offset,max_difference):
    """
    Associate two dictionaries of (stamp,data). As the time stamps never match exactly, we aim 
    to find the closest match for every input tuple.
    
    Input:
    first_list -- first dictionary of (stamp,data) tuples
    second_list -- second dictionary of (stamp,data) tuples
    offset -- time offset between both dictionaries (e.g., to model the delay between the sensors)
    max_difference -- search radius for candidate generation

    Output:
    matches -- list of matched tuples ((stamp1,data1),(stamp2,data2))
    
    """
    first_keys = first_list.keys()
    second_keys = second_list.keys()
    potential_matches = [(abs(a - (b + offset)), a, b) 
                         for a in first_keys 
                         for b in second_keys 
                         if abs(a - (b + offset)) < max_difference]
    potential_matches.sort()
    matches = []
    for diff, a, b in potential_matches:
        if a in first_keys and b in second_keys:
            first_keys = list(first_keys)
            first_keys.remove(a)
            second_keys = list(second_keys)
            second_keys.remove(b)
            matches.append((a, b))
    
    matches.sort()
    return matches

if __name__ == '__main__':
    main()
