import numpy as np
import cv2
import PIL
from sklearn.metrics import average_precision_score, roc_auc_score

# VOC: [x_min y_min x_max y_max]
# COCO: [x_min y_min width height]
# YOLO: [x_cen y_cen width height]

def downsize_boxes(boxes, width, height):
    boxes[..., [0, 2]] = boxes[..., [0, 2]] / width
    boxes[..., [1, 3]] = boxes[..., [1, 3]] / height
    return boxes

def upsize_boxes(boxes, width, height):
    boxes[..., [0, 2]] = boxes[..., [0, 2]] * width
    boxes[..., [1, 3]] = boxes[..., [1, 3]] * height
    return boxes

def coco2voc(boxes):
    boxes[..., 2] = boxes[..., 0] + boxes[..., 2]
    boxes[..., 3] = boxes[..., 1] + boxes[..., 3]
    return boxes

def voc2yolo(boxes):
    boxes[..., [2, 3]] = boxes[..., [2, 3]] - boxes[..., [0, 1]]
    boxes[..., [0, 1]] = boxes[..., [0, 1]] + boxes[..., [2, 3]] / 2
    return boxes

def yolo2voc(boxes):
    boxes[..., [0, 1]] = boxes[..., [0, 1]] - boxes[..., [2, 3]] / 2
    boxes[..., [2, 3]] = boxes[..., [2, 3]] + boxes[..., [0, 1]]
    return boxes

def np_metric_map_curve_by_class(probability, truth):
    num_sample, num_label = probability.shape
    score = []
    for i in range(num_label):
        s = average_precision_score(truth==i, probability[:,i])
        score.append(s)
    score = np.array(score)
    return score

