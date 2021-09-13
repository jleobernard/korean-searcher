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
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(nn.Linear(512, 1024), nn.ReLU(), nn.Linear(1024, 1))
        self.bb_regression = nn.Sequential(nn.Conv2d(kernel_size=(1, 1), in_channels=512, out_channels=6), nn.Sigmoid())

    def forward(self, x):
        x = self.features(x)
        # x.shape = _, 512, 13, 19
        x_class = nn.AdaptiveAvgPool2d((1, 1))(x)
        x_class = x_class.view(x.shape[0], -1)
        x_class = self.classifier(x_class)

        x_bb = self.bb_regression(x) # x_bb.shape = _, 6, 13, 19

        return x_class, x_bb


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

    def __init__(self, weights: [float], width: float, height: float, cell_width_stride: float = 19, cell_height_stride: float = 13):
        self.weights = weights
        self.epsilon = 1e-4
        self.width = width
        self.height = height
        self.cell_width = width / cell_width_stride
        self.cell_height = height / cell_height_stride

    def losses(self, out_classes, target_classes, out_bbs, target_bbs):
        # out_bbs.shape = _, 6, 13, 19
        # 6 => p_x0, x01, x02, p_x1, x11, x12
        loss_presence = F.binary_cross_entropy_with_logits(out_classes, target_classes.unsqueeze(1), reduction="sum")
        B, N, H, W = out_bbs.shape
        HxW = H * W
        preds = out_bbs.reshape(B, N, H * W)
        preds = preds.transpose(1, 2).contiguous() # B, H x W, N
        ########################
        ## Ys
        y_preds = preds[:, :, 0:3]
        maxes, indices = torch.max(y_preds[:, :, 0], dim=1)
        y_preds_with_boxes = torch.stack([y_preds[i, val, :] for i, val in enumerate(indices)])
        loss_confidence = ((1 - y_preds_with_boxes[:, 0]) ** 2).sum()
        designed_target_bbs_y = torch.stack([target_bbs[i, 0::2] * torch.tensor([int(i / self.cell_width) * self.cell_height, int(i / self.cell_width) * self.cell_height]) for i, val in enumerate(indices)])
        designed_target_bbs_y.requires_grad = False
        loss_y_corners = F.mse_loss(y_preds_with_boxes[:, 1:], designed_target_bbs_y)
        y_preds_without_boxes = torch.stack([torch.cat([y_preds[i, 0:val, :], y_preds[i, val + 1:, :]]) for i, val in enumerate(indices)], dim=0)
        loss_confidence += (y_preds_without_boxes[:, :, 0] ** 2).sum()

        x_preds = preds[:, :, 3:]

        #xs =
        center_y_hat = torch.mean(y_out_bbs, dim=1, keepdim=True)
        center_x_hat = torch.mean(x_out_bbs, dim=1, keepdim=True)
        center_x_gt  = torch.mean(x_target_bbs, dim=1, keepdim=True)
        center_y_gt  = torch.mean(y_target_bbs, dim=1, keepdim=True)
        loss_centers = (((center_x_hat - center_x_gt) ** 2 + (center_y_hat - center_y_gt) ** 2) * target_classes).sum()
        return loss_presence, loss_centers, loss_corners

    def aggregate_losses(self, losses):
        my_loss = 0
        for i in range(len(losses)):
            my_loss += losses[i] * self.weights[i]
        return my_loss

    def loss(self, out_classes, target_classes, out_bbs, target_bbs):
        curr_losses = self.losses(out_classes, target_classes, out_bbs, target_bbs)
        return self.aggregate_losses(curr_losses)

    def reshape_output(self, predictions):
        """
        Transforms the output of shape (B, F, H, W) into (B, H * W * F)
        :param predictions: Output of the network
        :return: Reshaped output
        """
        B, F, H, W = predictions.shape
        preds = predictions.reshape(B, F, H * W)
        preds = preds.transpose(1, 2).contiguous() # B, H x W, F
        preds = preds.reshape(B, H * W, 2, int(F / 2))
        return preds
