import os
import cv2
import glob
import numpy as np
from utils import make_folder
import multiprocessing as mp
import argparse


label_list = ['skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth', 'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']

folder_base = 'CelebAMask-HQ-mask-anno' # separate label folder
folder_save = 'CelebAMaskHQ-mask' # merged label save folder
img_num = 30000 # imgs total number

make_folder(folder_save) # make folder if not exist


def prepare_data(index_lowerBound, index_upperBound):
    for k in range(index_lowerBound, index_upperBound):
        folder_num = k // 2000
        im_base = np.zeros((512, 512))
        for idx, label in enumerate(label_list):
            filename = os.path.join(folder_base, str(folder_num), str(k).rjust(5, '0') + '_' + label + '.png')
            if (os.path.exists(filename)):
                print ('Exists ', filename, label, idx+1)
                im = cv2.imread(filename)
                im = im[:, :, 0]
                im_base[im != 0] = (idx + 1)
    
        filename_save = os.path.join(folder_save, str(k) + '.png')
        print (filename_save)
        cv2.imwrite(filename_save, im_base)


def mp_prepare(num_process):
    '''multi-processing for data preprocessing'''
    mp_cnt = num_process
    mp_list = []
    one_process_cnt = 30000 // mp_cnt
    for i in range(mp_cnt):
        index_lowerBound = i*one_process_cnt
        if i != mp_cnt-1:
            index_upperBound = (i+1)*one_process_cnt
        else:
            index_upperBound = 30000
        p = mp.Process(target=prepare_data, args=(index_lowerBound, index_upperBound))
        mp_list.append(p)
        p.start()

    for i in range(mp_cnt):
        p.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_process", type=int, default=8)
    args = parser.parse_args()
    print(args)
    num_process = args.num_process
    mp_prepare(num_process)
