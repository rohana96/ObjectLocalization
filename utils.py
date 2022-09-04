import os
import random
import time
import copy

import torch

import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np


# TODO: given bounding boxes and corresponding scores, perform non max suppression
def nms(bboxes, confidence_score, threshold=0.05):
    """
    bounding boxes of shape     Nx4
    confidence scores of shape  N
    threshold: confidence threshold for boxes to be considered

    return: list of bounding boxes and scores
    """
    bboxes = bboxes[confidence_score > threshold, :].cpu().detach().numpy()
    scores = confidence_score[confidence_score > threshold].cpu().detach().numpy()

    if bboxes.size == 0:
        return [], []

    ordered_indices = np.argsort(scores)[::-1]
    bboxes = bboxes[ordered_indices]
    scores = scores[ordered_indices]

    bboxes_after_nms = []
    scores_after_nms = []

    while bboxes.size != 0:
        chosen_box = bboxes[0]
        chosen_score = scores[0]

        keep_index = [
            i
            for i, box in enumerate(bboxes)
            if i != 0
            or iou(box, chosen_box) < 0.3
        ]

        bboxes = bboxes[keep_index]
        scores = scores[keep_index]

        bboxes_after_nms.append(chosen_box)
        scores_after_nms.append(chosen_score)

    return bboxes_after_nms, scores_after_nms


# TODO: calculate the intersection over union of two boxes
def iou(box1, box2):
    """
    Calculates Intersection over Union for two bounding boxes (xmin, ymin, xmax, ymax)
    returns IoU vallue
    """

    box1_x1, box1_y1, box1_x2, box1_y2 = box1
    box2_x1, box2_y1, box2_x2, box2_y2 = box2

    x1 = max(box1_x1, box2_x1)
    y1 = max(box1_y1, box2_y1)
    x2 = min(box1_x2, box2_x2)
    y2 = min(box1_y2, box2_y2)

    intersection = max((x2 - x1), 0) * (max((y2 - y1), 0))

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))
    union = max(1e-7, box1_area + box2_area - intersection)
    iou = intersection / union

    return iou


def tensor_to_PIL(image):
    """
    converts a tensor normalized image (imagenet mean & std) into a PIL RGB image
    will not work with batches (if batch size is 1, squeeze before using this)
    """
    inv_normalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.255],
    )

    inv_tensor = inv_normalize(image)
    inv_tensor = torch.clamp(inv_tensor, 0, 1)
    original_image = transforms.ToPILImage()(inv_tensor).convert("RGB")

    return original_image


def get_box_data(classes, bbox_coordinates, scores):
    """
    classes : tensor containing class predictions/gt
    bbox_coordinates: tensor containing [[xmin0, ymin0, xmax0, ymax0], [xmin1, ymin1, ...]] (Nx4)

    return list of boxes as expected by the wandb bbox plotter
    """
    box_list = [{
        "position": {
            "minX": bbox_coordinates[i][0],
            "minY": bbox_coordinates[i][1],
            "maxX": bbox_coordinates[i][2],
            "maxY": bbox_coordinates[i][3],
        },
        "class_id": classes[i],
        "scores": {
            "confidence": scores[i]
        }
    } for i in range(len(classes))
    ]

    return box_list
