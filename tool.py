from matplotlib import cm
import cv2
import numpy as np
from math import cos, sin, atan, pi
import torch


def draw_boxes(im, bboxes, is_display=True, color=None, caption="Image", wait=True):
    '''
        boxes: bounding boxes list. [l t r b] or
            [l_t r_t r_b l_b]
    '''
    if len(bboxes) == 0:
        return im
    is_poly = bboxes.shape[1] == 8 or bboxes.shape[1] == 9
    im = im.copy()


    if bboxes.shape[1] == 9 and np.all(np.abs(bboxes[:, -1])):
        sel_id = np.where(bboxes[:,-1] > 0)[0]

        bboxes = bboxes[sel_id, :8]

    for box in bboxes:
        if color == None:
            if len(box) == 5 or len(box) == 9:
                c = tuple(cm.jet([box[-1]])[0, 2::-1] * 255)
            else:
                c = tuple(np.random.randint(0, 256, 3))
        else:
            c = color
        if is_poly:
            cv2.polylines(im, np.int32(box[:8]).reshape(1, -1, 2), True, c, 2)
        else:
            cv2.rectangle(im, tuple(np.int32(box[:2])), tuple(np.int32(box[2:4])), c, 2)
    if is_display:
        cv2.imshow(caption, im)
        if wait:
            cv2.waitKey(0)
    return im

def rotate(xy, cxcy, theta):
    return (
        cos(theta) * (xy[0] - cxcy[0]) - sin(theta) * (xy[1] - cxcy[1]) + cxcy[0],
        sin(theta) * (xy[0] - cxcy[0]) + cos(theta) * (xy[1] - cxcy[1]) + cxcy[1]
    )


def poly_to_box2d(poly):
    """
    polys: 1*8
    """
    assert (len(poly) == 8)
    cx = (poly[0] + poly[4])/2
    cy = (poly[1] + poly[5])/2
    delta_y = poly[3] - poly[1]
    delta_x = poly[2] - poly[0]
    angle = atan(delta_y/(delta_x+0.0000001))

    box2d = np.zeros(5)
    x0, y0 = rotate((poly[0], poly[1]), (cx, cy), -angle)
    x1, y1 = rotate((poly[4], poly[5]), (cx, cy), -angle)
    w0 = abs(x1 - x0)
    h0 = abs(y1 - y0)
    box2d[...] = cx, cy, w0, h0, angle

    return box2d


def box2d_to_poly(box):
    """
    box: [cx, cy, w, h, angle]
    """
    cx = box[0]
    cy = box[1]
    w = box[2]
    h = box[3]
    angle = box[4]
    #print "angle", angle
    # xmin = int(cx - w/2.0)
    # xmax = int(cx + w/2.0)
    # ymin = int(cy - h/2.0)
    # ymax = int(cy + h/2.0)
    xmin = cx - w/2.0
    xmax = cx + w/2.0
    ymin = cy - h/2.0
    ymax = cy + h/2.0
    bb = np.zeros(8, np.float32)
    bb[:2] = rotate((xmin, ymin), (cx, cy), angle)
    bb[2:4] = rotate((xmax, ymin), (cx, cy), angle)
    bb[4:6] = rotate((xmax, ymax), (cx, cy), angle)
    bb[6:] = rotate((xmin, ymax), (cx, cy), angle)

    return bb

def hor_flip_boxes(gt_boxes, w):
    gt_boxes[:, :8:2] = w - gt_boxes[:, :8:2]
    return gt_boxes
    # assert (gt_boxes.shape[1] == 8)
    # gt_boxes_num = gt_boxes.shape[0]
    # fliped_boxes2d = np.zeros((gt_boxes_num, 5))
    # fliped_poly = np.zeros((gt_boxes_num, 8))
    # for n in range(gt_boxes_num):
    #     fliped_boxes2d[n] = poly_to_box2d(gt_boxes[n])
    #     fliped_boxes2d[n, 0] = w - fliped_boxes2d[n, 0]
    #     fliped_boxes2d[n, -1] = -fliped_boxes2d[n, -1]
    #     fliped_poly[n] = box2d_to_poly(fliped_boxes2d[n])
    #return fliped_poly


def rotate_boxes(im, gt_box, angle):
    rows, cols = im.shape[:2]

    # angle = np.random.randint(-90, 90)

    # angle = np.random.randint(-3, 3)

    M = cv2.getRotationMatrix2D((int(cols / 2), int(rows / 2)), angle, 1)

    dst = cv2.warpAffine(im, M, (cols, rows))
    dst = np.ascontiguousarray(dst, np.uint8)

    box_num = gt_box.shape[0]
    bb = np.zeros((box_num, 8), np.float32)

    for i in range(box_num):
        bb[i][:2] = rotate((gt_box[i][0], gt_box[i][1]), (int(cols / 2), int(rows / 2)), -angle / 180.0 * pi)
        bb[i][2:4] = rotate((gt_box[i][2], gt_box[i][3]), (int(cols / 2), int(rows / 2)), -angle / 180.0 * pi)
        bb[i][4:6] = rotate((gt_box[i][4], gt_box[i][5]), (int(cols / 2), int(rows / 2)), -angle / 180.0 * pi)
        bb[i][6:] = rotate((gt_box[i][6], gt_box[i][7]), (int(cols / 2), int(rows / 2)), -angle / 180.0 * pi)

    return dst, bb


def collate_fn(data):
    """Creates mini-batch tensors"""
    def pad_box(box, max_num):
        num = box.shape[0]
        assert box.shape[1] == 9
        padded_box = -2*torch.ones(max_num, 9)
        padded_box[:num] = box
        return padded_box

    boxes_num = [_['gt'].shape[0] for _ in data]
    max_box_num = max(boxes_num)
    images = torch.stack([_['image'] for _ in data])
    boxes = torch.stack([pad_box(_['gt'], max_box_num) for _ in data])

    return {'image': images, 'gt': boxes}




