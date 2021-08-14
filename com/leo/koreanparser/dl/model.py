from typing import Tuple

import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F

from com.leo.koreanparser.dl.utils.tensor_helper import to_best_device

class MyModel(nn.Module):

    def __init__(self):
        super(MyModel, self).__init__()
        resnet = models.resnet34(pretrained=False)
        for param in resnet.parameters():
            param.requires_grad = False
        layers = list(resnet.children())[:8]
        self.features1 = nn.Sequential(*layers[:6])
        self.features2 = nn.Sequential(*layers[6:])
        self.classifier = nn.Sequential(nn.BatchNorm1d(512), nn.Linear(512, 512), nn.Linear(512, 1))
        #self.classifier = nn.Linear(512, 1)
        self.bb = nn.Sequential(nn.BatchNorm1d(512), nn.Linear(512, 512), nn.Linear(512, 4))
        #self.bb = nn.Linear(512, 4)

    def forward(self, x):
        x = self.features1(x)
        x = self.features2(x)
        x = F.relu(x)
        x = nn.AdaptiveAvgPool2d((1, 1))(x)
        x = x.view(x.shape[0], -1)
        return self.classifier(x), self.bb(x)

    def initialize_weights(self):
        pass


def get_model(eval: bool = False):
    # load the model
    model = MyModel()
    # load the model onto the computation device
    if eval:
        model = model.eval()
    else:
        model = model.train()
    return to_best_device(model)


def model_loss(out_classes, target_classes, out_bbs, target_bbs, x):
    weights = to_best_device(torch.tensor([x.shape[-2] ** 2, x.shape[-1] ** 2, x.shape[-2] ** 2, x.shape[-1] ** 2], requires_grad=False))
    loss_class = F.binary_cross_entropy_with_logits(out_classes, target_classes.unsqueeze(1), reduction="sum")
    loss_bb = (F.mse_loss(out_bbs, target_bbs, reduction="none") / weights).sum()
    loss = loss_class + loss_bb * 4
    return loss
