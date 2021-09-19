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
        loss_presence = F.binary_cross_entropy_with_logits(out_classes, target_classes.unsqueeze(1), reduction="mean")
        # Reshape
        B, N, H, W = out_bbs.shape
        HxW = H * W
        preds = out_bbs.reshape(B, N, HxW)
        preds = preds.transpose(1, 2).contiguous() # B, H x W, N
        oneobj_hat = self.get_one_obj(preds)
        preds = preds.reshape(B * HxW, N)
        preds = torch.cat([preds[:, 1:3], preds[:, 4:]])
        cell_with_corners = self.get_cell_with_corners(target_bbs, height=H, width=W)
        oneobj_target = self.get_one_obj_target(target_bbs, cell_with_corners, height=H, width=W)
        loss_cell_presence = F.binary_cross_entropy_with_logits(oneobj_hat, oneobj_target, reduction="mean")
        reshaped_target_boxes = self.reshape_target_boxes(target_bbs, height=H, width=W)
        loss_distance_to_corners = (((preds - reshaped_target_boxes) ** 2).sum(dim=1) * oneobj_hat).mean()
        return loss_presence, loss_cell_presence, loss_distance_to_corners

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

    def get_one_obj(self, preds):
        """
        preds.shape = # B, H x W, N
        :return tensor of shape B * HxW * 2 with one where the corner was predicted
        """
        B,HxW, N = preds.shape
        size_coord = B * HxW
        oneobj = to_best_device(torch.zeros(size_coord * 2, requires_grad=False)) # x 2 because first ones are for ys and last ones for xs
        y_preds = preds[:, :, 0]
        _, indices = torch.max(y_preds, dim=1)
        for i, val in enumerate(indices):
            oneobj[i * HxW + val] = 1.
        x_preds = preds[:, :, 3]
        _, indices = torch.max(x_preds, dim=1)
        for i, val in enumerate(indices):
            oneobj[i * HxW + val + size_coord] = 1.

        return oneobj

    def get_one_obj_target(self, target_bbs, corners, height: int, width: int):
        """
        Transforme les target boxes en un grand one-hot vector de longueur B x H x W x 2.
        Il y a un 1 si la case est censée contenir un coin. Le coin en haut à gauche est contenu dans les B x H x W
        premières cellules. Le coin en bas à droite est contenu dans les B x H x W dernières cellules (d'où le x2 dans
        les dimensions).
        :param target_bbs: tensor de taille B x 4
        :param height: le nombde de cellules par image en hauteur
        :param width: le nombde de cellules par image en largeur
        :return: vecteur de taille B x H x W x 2
        """
        HxW = height * width
        B = target_bbs.shape[0]
        size_coord = B * HxW
        oneobj = to_best_device(torch.zeros(size_coord * 2, requires_grad=False))
        for i, val in enumerate(corners[:, 0]):
            oneobj[i * HxW + int(val)] = 1.
        for i, val in enumerate(corners[:, 1]):
            oneobj[i * HxW + int(val) + size_coord] = 1.
        return oneobj

    def reshape_target_boxes(self, target_bbs, height, width):
        """
        Transforme les target boxes (B, 4) en torch.tensor (B x H x W x 2, 2)
        Les B x H x W premières entrées contiennent les coordonnées du coin gauche supérieur et les B x H x W
        dernières les coordonnées du coin droit inférieur.
        Les coordonnées sont relatives au coin haut gauche de la cellule.
        :param target_bbs: Targets boxes initiales
        :param height: Nombre de cellules par hauteur
        :param width: Nombre de cellules par largeur
        :return:
        """
        B, _ = target_bbs.shape
        HxW = height * width
        targets = torch.cat([coord.expand(HxW, -1) for coord in torch.cat([target_bbs[:, 0:2], target_bbs[:, 2:]])])
        template_origin = torch.zeros((HxW, 2), requires_grad=False)
        idx = 0
        for i in range(height):
            ii = i / height
            for j in range(width):
                template_origin[idx] = torch.tensor([ii, j / width], requires_grad=False)
                idx += 1
        template_origin = template_origin.repeat(B * 2, 1)
        targets = (targets - template_origin) * torch.tensor([height, width], requires_grad=False)
        return targets



    def get_cell_with_corners(self, target_bbs, height, width):
        new_tbs = torch.floor(target_bbs * torch.tensor([height, width, height, width], requires_grad=False))
        corners = torch.cat([
            (new_tbs[:, 0] * width + new_tbs[:, 1]).unsqueeze(0),
            (new_tbs[:, 2] * width + new_tbs[:, 3]).unsqueeze(0)
        ])
        return corners


"""

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
"""
