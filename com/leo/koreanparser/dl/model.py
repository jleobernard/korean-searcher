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
        self.lstm_h = nn.LSTM(input_size=512, hidden_size=512, num_layers=2, batch_first=True, bidirectional=True)
        self.lstm_w = nn.LSTM(input_size=512, hidden_size=512, num_layers=2, batch_first=True, bidirectional=True)
        self.classifier_h = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(), nn.Linear(512, 2), nn.Sigmoid())
        self.classifier_w = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(), nn.Linear(512, 2), nn.Sigmoid())

    def forward(self, x):
        x = self.features(x)
        # x.shape = _, 512, 13, 19
        x_class = nn.AdaptiveAvgPool2d((1, 1))(x)
        x_class = x_class.view(x.shape[0], -1)
        x_class = self.classifier(x_class)
        ###################
        ## Width
        x_width = torch.sum(x, 2) # x_width.shape = _, 512, 19
        x_width = torch.sigmoid(x_width)
        x_width = torch.transpose(x_width, 1, 2)# x_width.shape = _, 19, 512
        out_x_width, _ = self.lstm_w(x_width)
        out_x_width = torch.cat((out_x_width[:, 0, :], out_x_width[:, -1, :]), 1)
        out_x_width = self.classifier_w(out_x_width)
        ###################
        ## Height
        x_height = torch.sum(x, 3) # x_height.shape = _, 512, 13
        x_height = torch.sigmoid(x_height)
        x_height = torch.transpose(x_height, 1, 2) # x_height.shape = _, 13, 512
        out_x_height, _ = self.lstm_w(x_height)
        out_x_height = torch.cat((out_x_height[:, 0, :], out_x_height[:, -1, :]), 1)
        out_x_height = self.classifier_h(out_x_height)

        return x_class, (out_x_height, out_x_width)


    def initialize_weights(self):
        modules = [self.lstm_w, self.lstm_h]
        for m in modules:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)


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

    def __init__(self, weights: [float]):
        self.weights = weights
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
        y_out_bbs, x_out_bbs = out_bbs
        y_target_bbs = target_bbs[:, [0, 2]]
        x_target_bbs = target_bbs[:, [1, 3]]
        loss_class = F.binary_cross_entropy_with_logits(out_classes, target_classes.unsqueeze(1), reduction="sum")
        mse1 = F.mse_loss(y_out_bbs, y_target_bbs, reduction="none").sum(dim=1)
        mse2 = F.mse_loss(x_out_bbs, x_target_bbs, reduction="none").sum(dim=1)
        print(f"mse1 shape is {mse1.shape}")
        print(f"mse2 shape is {mse2.shape}")
        print(f"target_classes shape shape is {target_classes.shape}")
        loss_corners = ((mse1 + mse2) * target_classes).sum()
        center_y_hat = torch.mean(y_out_bbs, dim=1, keepdim=True)
        center_x_hat = torch.mean(x_out_bbs, dim=1, keepdim=True)
        center_x_gt  = torch.mean(x_target_bbs, dim=1, keepdim=True)
        center_y_gt  = torch.mean(y_target_bbs, dim=1, keepdim=True)
        loss_centers = (((center_x_hat - center_x_gt) ** 2 + (center_y_hat - center_y_gt) ** 2) * target_classes).sum()
        return loss_class, loss_centers, loss_corners

    def aggregate_losses(self, losses):
        my_loss = 0
        for i in range(len(losses)):
            my_loss += losses[i] * self.weights[i]
        return my_loss

    def loss(self, out_classes, target_classes, out_bbs, target_bbs):
        curr_losses = self.losses(out_classes, target_classes, out_bbs, target_bbs)
        return self.aggregate_losses(curr_losses)
