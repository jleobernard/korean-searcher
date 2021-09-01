import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


from com.leo.koreanparser.dl.utils.tensor_helper import to_best_device

class MyModel(nn.Module):

    def __init__(self):
        super(MyModel, self).__init__()
        resnet = models.resnet34(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad = True
        layers = list(resnet.children())[:8]
        self.features1 = nn.Sequential(*layers[:6])
        self.features2 = nn.Sequential(*layers[6:])
        self.classifier = nn.Sequential(nn.BatchNorm1d(512), nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 1))
        self.bb = nn.Sequential(nn.BatchNorm1d(512), nn.Linear(512, 256), nn.ReLU(),
                                nn.Dropout(p=0.22),
                                nn.Linear(256, 128), nn.ReLU(),
                                nn.BatchNorm1d(128),
                                nn.Linear(128, 4), nn.ReLU())

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

class ModelLoss:

    def __init__(self, weights: [float], constant_width: float = 400):
        self.weights = weights
        self.constant_width = constant_width
        self.normalization_factor = constant_width ** 2
        self.epsilon = 1e-4

    def iou(self, out, target):
        """
        :param out: Tensor of shape (B, 4)
        :param target:  Tensor of shape (B, 4)
        :return:  Tensor of shape (B, 1)
        """
        area_out = ((out[:, 2]-out[:, 0]) * (out[:, 3]-out[:, 1])) + self.epsilon
        area_target = ((target[:, 2]-target[:, 0]) * (target[:, 3]-target[:, 1])) + self.epsilon

        lt = torch.max(out[:, :2], target[:, :2])  # [rows, 2]
        rb = torch.min(out[:, 2:], target[:, 2:])  # [rows, 2]

        wh = (rb - lt).clamp(min=0)  # [rows, 2]
        inter = wh[:, 0] * wh[:, 1]

        return (0.5 - (inter / (area_target + area_out - inter))).sum()

    def losses(self, out_classes, target_classes, out_bbs, target_bbs):
        loss_class = F.binary_cross_entropy_with_logits(out_classes, target_classes.unsqueeze(1), reduction="sum")
        loss_corners = F.mse_loss(out_bbs, target_bbs, reduction="sum")
        center_x_hat = (out_bbs[:, 2] + out_bbs[:, 0]) / 2
        center_x_gt  = (target_bbs[:, 2] + target_bbs[:, 0]) / 2
        center_y_hat = (out_bbs[:, 3] + out_bbs[:, 1]) / 2
        center_y_gt  = (target_bbs[:, 3] + target_bbs[:, 1]) / 2
        loss_centers = ((center_x_hat - center_x_gt) ** 2 + (center_y_hat - center_y_gt) ** 2).sum()
        """
        out_bbs = out_bbs / self.constant_width
        target_bbs = target_bbs / self.constant_width
        longueur_gt = (target_bbs[:, 2] - target_bbs[:, 0]) + 1 # Pour éviter les divisions par zéro
        largeur_gt = (target_bbs[:, 3] - target_bbs[:, 1]) + 1
        longueur_hat = (out_bbs[:, 2] - out_bbs[:, 0]) + 1
        largeur_hat = (out_bbs[:, 3] - out_bbs[:, 1]) + 1
        d1gt = target_bbs[:, 0] + longueur_gt / 2
        d2gt = target_bbs[:, 1] + largeur_gt / 2
        d1 = out_bbs[:, 0] + longueur_hat / 2
        d2 = out_bbs[:, 1] + largeur_hat / 2
        loss_dc = ((d1gt - d1) ** 2 + (d2gt - d2) ** 2)
        loss_dc = loss_dc.sum()
        loss_ratio = (((longueur_gt / largeur_gt) - (longueur_hat / largeur_hat)) ** 2)
        loss_ratio = loss_ratio.sum()
        loss_diff = ((longueur_gt - longueur_hat) ** 2)
        loss_diff = loss_diff.sum()
        return loss_class, loss_dc, loss_ratio, loss_diff
        """
        return loss_class, loss_centers, loss_corners

    def aggregate_losses(self, losses):
        my_loss = 0
        for i in range(len(losses)):
            my_loss += losses[i] * self.weights[i]
        return my_loss

    def loss(self, out_classes, target_classes, out_bbs, target_bbs):
        curr_losses = self.losses(out_classes, target_classes, out_bbs, target_bbs)
        return self.aggregate_losses(curr_losses)
