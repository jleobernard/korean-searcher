import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

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
        self.classifier = nn.Sequential(nn.BatchNorm1d(512), nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 1))
        self.bb = nn.Sequential(nn.BatchNorm1d(512), nn.Linear(512, 256), nn.ReLU(),
                                nn.Dropout(), nn.Linear(256, 128), nn.ReLU(),
                                nn.BatchNorm1d(128), nn.Linear(128, 4), nn.Sigmoid())

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

    def intersect(self, box_a, box_b):
        """ We resize both tensors to [A,B,2] without new malloc:
        [A,2] -> [A,1,2] -> [A,B,2]
        [B,2] -> [1,B,2] -> [A,B,2]
        Then we compute the area of intersect between box_a and box_b.
        Args:
          box_a: (tensor) bounding boxes, Shape: [A,4].
          box_b: (tensor) bounding boxes, Shape: [B,4].
        Return:
          (tensor) intersection area, Shape: [A,B].
        """
        A = box_a.size(0)
        B = box_b.size(0)
        max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                           box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
        min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                           box_b[:, :2].unsqueeze(0).expand(A, B, 2))
        inter = torch.clamp((max_xy - min_xy), min=0)
        return inter[:, :, 0] * inter[:, :, 1]

    def jaccard(self, box_a, box_b):
        """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
        is simply the intersection over union of two boxes.  Here we operate on
        ground truth boxes and default boxes.
        E.g.:
            A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
        Args:
            box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
            box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
        Return:
            jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
        """
        inter = self.intersect(box_a, box_b)
        area_a = ((box_a[:, 2]-box_a[:, 0]) *
                  (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
        area_b = ((box_b[:, 2]-box_b[:, 0]) *
                  (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
        union = area_a + area_b - inter
        return inter / union  # [A,B]

    def losses(self, out_classes, target_classes, out_bbs, target_bbs):
        loss_class = F.binary_cross_entropy_with_logits(out_classes, target_classes.unsqueeze(1), reduction="sum")
        loss_bbs = F.mse_loss(out_bbs, target_bbs, reduction="sum")
        center_x_hat = (out_bbs[:, 2] + out_bbs[:, 0]) / 2
        center_x_gt  = (target_bbs[:, 2] + target_bbs[:, 0]) / 2
        center_y_hat = (out_bbs[:, 3] + out_bbs[:, 1]) / 2
        center_y_gt  = (target_bbs[:, 3] + target_bbs[:, 1]) / 2
        loss_centers = ((center_x_hat - center_x_gt) ** 2 + (center_y_hat - center_y_gt) ** 2).sum()
        loss_iou = self.jaccard(target_bbs, out_bbs).sum()
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
        return loss_class, loss_bbs, loss_centers, loss_iou

    def aggregate_losses(self, losses):
        my_loss = 0
        for i in range(len(losses)):
            my_loss += losses[i] * self.weights[i]
        return my_loss

    def loss(self, out_classes, target_classes, out_bbs, target_bbs):
        curr_losses = self.losses(out_classes, target_classes, out_bbs, target_bbs)
        return self.aggregate_losses(curr_losses)
