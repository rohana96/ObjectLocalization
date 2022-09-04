import torch
import torchvision
import torch.utils.data as data
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from AlexNet import LocalizerAlexNet
import numpy as np

from torchvision.ops import roi_pool, roi_align


class WSDDN(nn.Module):
    n_classes = 20
    classes = np.asarray([
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
        'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ])

    def __init__(self, classes=None):
        super(WSDDN, self).__init__()

        if classes is not None:
            self.classes = classes
            self.n_classes = len(classes)
            print(classes)

        alexnet = LocalizerAlexNet()
        # TODO: Define the WSDDN model
        self.features = alexnet.features

        # self.roi_pool = None

        self.classifier = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
        )

        self.score_fc = nn.Linear(4096, self.n_classes)
        self.bbox_fc = nn.Linear(4096, self.n_classes)
        # # loss
        self.cross_entropy = None #nn.BCELoss()

    @property
    def loss(self):
        return self.cross_entropy

    def forward(self,
                image,
                rois=None,
                gt_vec=None,
                ):

        # TODO: Use image and rois as input
        # compute cls_prob which are N_roi X 20 scores
        # rois = rois * image.shape[-1]  # 512
        out = self.features(image)
        rois_list = [roi.type(torch.float) for roi in rois]
        rois = roi_pool(out, rois_list, output_size = (6, 6), spatial_scale=31.0/512)
        rois = rois.view(-1, 256 * 6 * 6)
        out = self.classifier(rois)

        class_score = self.score_fc(out)
        region_score = self.bbox_fc(out)

        class_score = torch.softmax(class_score, dim = -1)
        region_score = torch.softmax(region_score, dim = 0)
        cls_prob = class_score * region_score

        if self.training:
            label_vec = gt_vec.view(self.n_classes, -1)
            self.cross_entropy = self.build_loss(cls_prob, label_vec)

        return cls_prob

    def build_loss(self, cls_prob, label_vec):
        """Computes the loss

        :cls_prob: N_roix20 output scores
        :label_vec: 1x20 one hot label vector
        :returns: loss

        """
        # TODO: Compute the appropriate loss using the cls_prob that is the
        # output of forward()
        # Checkout forward() to see how it is called

        cls_prob = torch.clamp(torch.sum(cls_prob, dim=0), min=0.0, max=1.0)
        loss = nn.BCELoss(reduction="sum")(cls_prob, label_vec.squeeze())
        return loss
