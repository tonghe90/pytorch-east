from __future__ import print_function
import torch
import numpy as np
from tsfrm import Tsfrm_random_crop, Tsfrm_scale, Tsrfm_random_scale, Tsrfm_sub_mean, Tsrfm_to_tensor, \
                   Tsrfm_random_flip, Tsrfm_random_rotate, Tsfrm_random_crop_text

from tool import draw_boxes, collate_fn
from data_provider import ICDAR15_Dataset
from torchvision import transforms
from torch.utils.data import DataLoader

##### data_layer

min_ratio = 0.8
max_ratio = 1.8
batch_size = 1
transforms = transforms.Compose([Tsrfm_random_scale(min_ratio, max_ratio),
                                 Tsrfm_random_rotate(-20, 20),
                                 Tsrfm_sub_mean(122.0),
                                 Tsfrm_random_crop_text(960),
                                 Tsrfm_random_flip(),
                                 Tsrfm_to_tensor()])

icdar15_train = ICDAR15_Dataset(images_dir="/home/tonghe/DATA/icdar15/train/imgs/",
                                gt_dir="/home/tonghe/DATA/icdar15/train/gts/",
                                transform=transforms)

dataloader_train = DataLoader(icdar15_train, batch_size=batch_size,
                        shuffle=True, num_workers=1, collate_fn=collate_fn)

# for i, sample_batched in enumerate(dataloader_train):
#     images = sample_batched['image']
#     gt_boxes = sample_batched['gt']
#     print(i, ' / ', len(dataloader_train), images.size(), gt_boxes.size())
#     if i == 3:
#         break
#     images = np.uint8(images.numpy() + 122)
#     gt_boxes = gt_boxes.numpy()
#
#     sel_id = np.where(gt_boxes[0][:,-1]>0)[0]
#     for n in range(batch_size):
#        draw_boxes(images[n].transpose(1,2,0), gt_boxes[n])




