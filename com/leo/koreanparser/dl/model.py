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
        self.alpha, self.beta, self.gamma, self.theta = weights
        self.constant_width = constant_width
        self.normalization_factor = constant_width ** 2

    def losses(self, out_classes, target_classes, out_bbs, target_bbs):
        loss_class = F.binary_cross_entropy_with_logits(out_classes, target_classes.unsqueeze(1), reduction="sum")
        loss_bbs = F.mse_loss(out_bbs, target_bbs, reduction="sum")
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
        return loss_class, loss_bbs

    def aggregate_losses(self, losses):
        loss_class, loss_bbs = losses
        my_loss = self.alpha * loss_class + self.beta * loss_bbs
        return my_loss

    def loss(self, out_classes, target_classes, out_bbs, target_bbs):
        loss_class, loss_bbs = self.losses(out_classes, target_classes, out_bbs, target_bbs)
        my_loss = self.alpha * loss_class + self.beta * loss_bbs
        return my_loss
