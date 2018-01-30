import torch
import numpy as np
from torchvision import transforms
import cv2
from tool import hor_flip_boxes, rotate_boxes
from tool import draw_boxes
from numpy.random import randint


class Tsfrm_scale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, gt_boxes = sample['image'], sample['gt']
        assert gt_boxes.shape[1] == 9
        if np.all(gt_boxes[:,-1] == -1):
            return {'image': image, 'gt': gt_boxes}
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size, float(self.output_size) * w / h
            else:
                new_h, new_w = float(self.output_size * h) / w, self.output_size
        else:
            new_h, new_w = self.output_size

        #image = transforms.Resize(image, (new_h, new_w))

        dst_image = cv2.resize(image, (int(new_w), int(new_h)))
        box_ids = np.where(gt_boxes[:,-1] > 0)[0]
        tmp = gt_boxes[[box_ids],:8].reshape(-1,4,2) * [new_w / float(w), new_h / float(h)]
        gt_boxes[box_ids, :8] = tmp
        #boxes = np.concatenate((tmp.reshape(-1,8), gt_boxes[:,[-1]]), axis=1)
        return {'image': dst_image, 'gt': gt_boxes}


class Tsfrm_random_crop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, gt_boxes = sample['image'], sample['gt']
        assert gt_boxes.shape[-1] == 9
        if np.all(gt_boxes[:,-1] == -1):
            return {'image': image, 'gt': gt_boxes}
        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        crop_image = np.zeros((new_h, new_w, image.shape[2]))
        if new_h >= h:
            start_h = 0
            end_h = h
        else:
            start_h = np.random.randint(0, h - new_h)
            end_h = start_h + new_h

        if new_w >= w:
            start_w = 0
            end_w = w
        else:
            start_w = np.random.randint(0, w - new_w)
            end_w = start_w + new_w

        gh = int(end_h-start_h)
        gw = int(end_w-start_w)
        crop_image[:gh, :gw, :] = image[start_h:end_h, start_w:end_w, :]
        box_ids = np.where(gt_boxes[:,-1] > 0)[0]
        tmp = gt_boxes[box_ids,:8].reshape(-1,4,2) - [start_w, start_h]
        gt_boxes[box_ids, :8] = tmp
        return {'image': crop_image, 'gt': gt_boxes}




class Tsfrm_random_crop_text(object):
    def __init__(self, output_size):
        assert isinstance(output_size, int)
        self.output_size = int(output_size)

    def __call__(self, sample):
        image, gt_boxes = sample['image'], sample['gt']
        if np.all(gt_boxes[:,-1] == -1):
            return {'image': image, 'gt': gt_boxes}
        assert gt_boxes.shape[1] == 9
        box_ids = np.where(gt_boxes[:,-1] > 0)[0]
        resized_im, w_off, h_off = self.crop_image_label(image, gt_boxes[box_ids], self.output_size)

        crop_image = resized_im[h_off:h_off + self.output_size, w_off: w_off + self.output_size, :]
        gt_boxes[box_ids, :8:2] = gt_boxes[box_ids, :8:2] - w_off
        gt_boxes[box_ids, 1:8:2] = gt_boxes[box_ids, 1:8:2] - h_off

        gt_num = gt_boxes.shape[0]
        for k in range(gt_num):
            xs = gt_boxes[k, :8:2]
            ys = gt_boxes[k, 1:8:2]
            if gt_boxes[k, -1] < 0:
                continue
            if max(xs) <= 0 or min(xs) >= self.output_size or \
               max(ys) <= 0 or min(ys) >= self.output_size or \
               max(xs) >= self.output_size or min(xs) < 0 or \
               max(ys) >= self.output_size or min(ys) < 0:
                gt_boxes[k,-1] = -1
        #

        return {'image': crop_image, 'gt': gt_boxes}

    def crop_image_label(self, image, gt_boxes, crop_size):
        im_h, im_w = image.shape[:2]
        pad_h = max(0, crop_size - im_h + 2)
        pad_w = max(0, crop_size - im_w + 2)
        if pad_h > 0 or pad_w > 0:
            image = cv2.copyMakeBorder(image, 0, pad_h,
                                       0, pad_w, cv2.BORDER_CONSTANT, (122.0, 122.0, 122.0))

        im_h, im_w = image.shape[:2]
        gt_num = gt_boxes.shape[0]
        print 'gt_boxes2:', gt_boxes
        sel_id = int(randint(0, gt_num, 1))
        h_middle = max(int(np.sum(gt_boxes[sel_id][1::2]) / 4.0), 1)
        w_middle = max(int(np.sum(gt_boxes[sel_id][::2]) / 4.0), 1)

        w_left = max(w_middle - crop_size, 0)
        w_right = min(w_middle, im_w - crop_size)
        h_top = max(h_middle - crop_size, 0)
        h_bottom = min(h_middle, im_h - crop_size)

        if (h_bottom - h_top) <= 0 or (w_right - w_left) <= 0:
            return image, 0, 0

        assert (h_bottom - h_top) > 0, 'h_bottom: {}, h_top: {}, h_middle: {}, im_h: {}'.format(h_bottom, h_top,
                                                                                                h_middle, im_h)
        assert (w_right - w_left) > 0, 'w_right: {}, w_left: {}, w_middle:{}, im_w:{}'.format(w_right, w_left, w_middle,
                                                                                              im_w)
        h_off = randint(h_top, h_bottom, 1)
        w_off = randint(w_left, w_right, 1)
        return image, int(w_off), int(h_off)




class Tsrfm_random_scale(object):
    def __init__(self, min_ratio, max_ratio):
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def __call__(self, sample):
        scale = np.random.randint(int(self.min_ratio * 10000), int(self.max_ratio * 10000)) / float(10000.0)
        image, gt_boxes = sample['image'], sample['gt']
        if np.all(gt_boxes[:,-1] == -1):
            return {'image': image, 'gt': gt_boxes}
        assert gt_boxes.shape[-1] == 9
        h, w = image.shape[:2]
        new_h = int(scale * h)
        new_w = int(scale * w)

        image = cv2.resize(image, (new_w, new_h))
        box_ids = np.where(gt_boxes[:, -1] > 0)[0]
        tmp = gt_boxes[box_ids,:8] * scale
        #gt_boxes = np.concatenate((tmp, gt_boxes[:,[8]]), axis=1)
        gt_boxes[box_ids, :8] = tmp
        return {'image': image, 'gt': gt_boxes}

class Tsrfm_sub_mean(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, mean_val):
        self.mean_val = mean_val

    def __call__(self, sample):
        image, gt_boxes = sample['image'], sample['gt']
        image = image - self.mean_val

        return {'image': image,
                'gt': gt_boxes}


class Tsrfm_random_flip(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self):
        pass

    def __call__(self, sample):
        image, gt_boxes = sample['image'], sample['gt']
        assert gt_boxes.shape[1] == 9
        if np.all(gt_boxes[:,-1] == -1):
            return {'image': image, 'gt': gt_boxes}
        tmp = np.random.rand(1)
        w = image.shape[1]
        box_ids = np.where(gt_boxes[:, -1] > 0)[0]
        if tmp > 0.5:
            image = image[:, ::-1, :]
            gt_boxes[box_ids,:8] = hor_flip_boxes(gt_boxes[box_ids,:8], w)

        return {'image': image,
                'gt': gt_boxes}


class Tsrfm_random_rotate(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, min_angle, max_angle):
        self.min_angle = min_angle
        self.max_angle = max_angle

    def __call__(self, sample):
        image, gt_boxes = sample['image'], sample['gt']
        assert gt_boxes.shape[1] == 9
        if np.all(gt_boxes[:,-1] == -1):
            return {'image': image, 'gt': gt_boxes}
        angle = np.random.randint(self.min_angle, self.max_angle, 1)[0]
        box_ids = np.where(gt_boxes[:, -1] > 0)[0]
        image, boxes = rotate_boxes(image, gt_boxes[box_ids,:8], angle)
        gt_boxes[box_ids, :8] = boxes

        return {'image': image,
                'gt': gt_boxes}


class Tsrfm_to_tensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, gt_boxes = sample['image'], sample['gt']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        image = np.float32(image)
        gt_boxes = np.float32(gt_boxes)

        return {'image': torch.from_numpy(image),
                'gt': torch.from_numpy(gt_boxes)}



