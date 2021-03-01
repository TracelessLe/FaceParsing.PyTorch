import os.path as osp
import torch
import timeit
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
# from torchvision import transforms

from networks import get_model
from utils import *
# from PIL import Image
import time
import cv2
from metrics import SegMetric


# def make_dataset(dir):
#     images = []
#     assert osp.isdir(dir), '%s is not a valid directory' % dir

#     f = dir.split('/')[-1].split('_')[-1]
#     print(dir, len([name for name in os.listdir(dir)
#                     if osp.isfile(osp.join(dir, name))]))
#     for i in range(len([name for name in os.listdir(dir) if osp.isfile(osp.join(dir, name))])):
#         img = str(i) + '.jpg'
#         path = osp.join(dir, img)
#         images.append(path)

#     return images


class Tester(object):

    def __init__(self, data_loader, config):

        # Data loader
        self.data_loader = data_loader

        # Model hyper-parameters
        self.imsize = config.imsize
        self.parallel = config.parallel
        self.classes = config.classes
        self.pretrained_model = config.pretrained_model  # int type

        self.model_save_path = config.model_save_path
        self.arch = config.arch
        # self.test_size = config.test_size
        self.batch_size = config.batch_size
        self.test_colorful = config.test_colorful
        self.test_color_label_path = osp.join(config.test_color_label_path, self.arch)
        self.test_pred_label_path = osp.join(config.test_pred_label_path, self.arch)

        self.build_model()

    def test(self):

        time_meter = AverageMeter()

        # Model loading
        self.G.load_state_dict(torch.load(
            osp.join(self.model_save_path, self.arch, "{}_G.pth".format(self.pretrained_model))))
        self.G.eval()
        # batch_num = int(self.test_size / self.batch_size)
        metrics = SegMetric(n_classes=self.classes)
        metrics.reset()
        
        index = 0
        for index, (images, labels) in enumerate(self.data_loader):
            print('processing batch %d' % (index))
            if (index + 1) % 100 == 0:
                print('%d batches processd' % (index + 1))

            images = images.cuda()
            labels = labels.cuda()
            size = labels.size()
            h, w = size[1], size[2]

            torch.cuda.synchronize()
            tic = time.perf_counter()

            with torch.no_grad():
                outputs = self.G(images)
                # Whether or not multi branch?
                if self.arch == 'CE2P' or 'FaceParseNet' in self.arch:
                    outputs = outputs[0][-1]

                outputs = F.interpolate(outputs, (h, w), mode='bilinear', align_corners=True)
                pred = outputs.data.max(1)[1].cpu().numpy()  # Matrix index
                gt = labels.cpu().numpy()
                metrics.update(gt, pred)

            torch.cuda.synchronize()
            time_meter.update(time.perf_counter() - tic)

            if self.test_colorful: # Whether color the test results to png files
                # labels_predict_color = generate_label(outputs, self.imsize)
                labels = labels[:, :, :].view(size[0], 1, size[1], size[2])
                oneHot_size = (size[0], self.classes, size[1], size[2])
                labels_real = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
                labels_real = labels_real.scatter_(1, labels.data.long().cuda(), 1.0)
                labels_predict_plain = generate_label_plain(outputs, self.imsize)
                compare_predict_color = generate_compare_results(images, labels_real, outputs, self.imsize)
                for k in range(self.batch_size):
                    # save_image(labels_predict_color[k], osp.join(self.test_color_label_path, str(index * self.batch_size + k) +'.png'))
                    cv2.imwrite(osp.join(self.test_pred_label_path, str(index * self.batch_size + k) +'.png'), labels_predict_plain[k])
                    save_image(compare_predict_color[k], osp.join(self.test_color_label_path, str(index * self.batch_size + k) +'.png'))

        print("----------------- Runtime Performance ------------------")
        print('Total %d batches (%d images) tested.' % (index + 1, (index+1)*images.size(0)))
        print("Inference Time per image: {:.4f}s".format(time_meter.average() / images.size(0)))
        print("Inference FPS: {:.2f}".format(images.size(0) / time_meter.average()))

        score = metrics.get_scores()[0]
        class_iou = metrics.get_scores()[1]

        print("----------------- Total Performance --------------------")
        for k, v in score.items():
            print(k, v)

        print("----------------- Class IoU Performance ----------------")
        facial_names = ['background', 'skin', 'nose', 'eyeglass', 'left_eye', 'right_eye', 'left_brow', 'right_brow',
                        'left_ear', 'right_ear', 'mouth', 'upper_lip', 'lower_lip', 'hair', 'hat', 'earring', 'necklace',
                        'neck', 'cloth']
        for i in range(self.classes):
            print(facial_names[i] + "\t: {}".format(str(class_iou[i])))
        print("--------------------------------------------------------")


    def build_model(self):
        self.G = get_model(self.arch, pretrained=False).cuda()
        if self.parallel:
            self.G = nn.DataParallel(self.G)
