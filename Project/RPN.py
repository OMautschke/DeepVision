import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


class RPN(nn.Module):

    def __init__(self, in_channel, mid_channel, ratio=[0.5, 1, 2], anchor_size=[128, 256, 512]):
        super(RPN, self).__init__()

        self.ratio = ratio
        self.anchor_size = anchor_size
        self.k = len(ratio) * len(anchor_size)

        self.rpn_conv = nn.Sequential(nn.Conv2d(in_channel, mid_channel, kernel_size=5, stride=1, padding=0),
                                       nn.ReLU())

        # 9 anchor * 2 classfier (object or non-object) each grid
        self.class_score = nn.Conv2d(mid_channel, 2 * 9, kernel_size=1, stride=1)

        # 9 anchor * 4 coordinate regressor each grids
        self.box_pred = nn.Conv2d(mid_channel, 4 * 9, kernel_size=1, stride=1)
        self.softmax = nn.Softmax()

    def forward(self, features):
        features = self.rpn_conv(features)
        #ligits, rpn_bbox_pred = self.conv1(features), self.conv2(features)

        _, _, height, width = features.shape

        class_score = self.class_score(features)
        box_pred = self.box_pred(features)

        anchor = []

        return class_score, box_pred, anchor
