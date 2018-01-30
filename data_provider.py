from __future__ import print_function
import os
import os.path as osp
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import cv2

from torchvision import transforms
from tool import draw_boxes

class ICDAR15_Dataset(Dataset):
    def __init__(self, images_dir, gt_dir, transform=None):
        self.images_dir = images_dir
        self.gt_dir = gt_dir
        self.im_names = sorted(os.listdir(self.images_dir))
        self.transform = transform

    def __len__(self):
        return len(self.im_names)

    def skip_BOM(self, fp):
        str = fp.read(3)
        if str != '\xef\xbb\xbf':
            fp.seek(0)

    def parse_gt(self, im_file):
        #final_boxes = -np.ones((max_box_num, 9), dtype=np.float32)
        boxes = []
        ignore_label = []
        num = 0
        with open(im_file) as fp:
            self.skip_BOM(fp)
            for line in fp:
                line = line.strip()
                if line != "":
                    parts = line.split(",")

                    box = map(lambda x: int(x), parts[:8])
                    text = ",".join(parts[8:])
                    boxes.append(box)
                    num += 1
                    if text != "###":
                        ignore_label.append(1)
                    else:
                        ignore_label.append(-1)
                        # boxes.append(box)

        boxes = np.hstack((np.float32(boxes).reshape(num, 8), np.float32(ignore_label).reshape(-1, 1)))
        # if boxes.shape[0] > max_box_num:
        #     sel_id = np.int32(np.random.choice(np.arange(boxes.shape[0]), max_box_num, replace=False))
        #     final_boxes = boxes[sel_id]
        # else:
        #     final_boxes[:boxes.shape[0]] = boxes
        #
        # return final_boxes
        return boxes

    def __getitem__(self, item):
        img_name = osp.join(self.images_dir, self.im_names[item])
        print('img_name: ', img_name, '***', item)
        image = cv2.imread(img_name)
        #image = Image.open(img_name)
        gt_file = osp.join(self.gt_dir, "gt_" + osp.splitext(self.im_names[item])[0] + ".txt")
        gts = self.parse_gt(gt_file)

        sample = {'image': image, 'gt': gts}

        if self.transform:
            sample = self.transform(sample)

        return sample




if __name__ == '__main__':
    icdar15_train = ICDAR15_Dataset(images_dir="/home/tonghe/DATA/icdar15/train/imgs/",
                                    gt_dir="/home/tonghe/DATA/icdar15/train/gts/")
    print(len(icdar15_train))
    for i in range(len(icdar15_train)):
        sample = icdar15_train[i]

        image = sample['image']
        gt_boxes = sample['gt']

        print(i, image.shape, gt_boxes.shape)
        draw_boxes(image, gt_boxes)
