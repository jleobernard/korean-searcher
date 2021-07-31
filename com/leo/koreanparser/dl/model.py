import torchvision
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from com.leo.koreanparser.dl.utils.tensor_helper import to_best_device


class MyModel(nn.Module):

    def ___init__(self):
        super(MyModel, self).__init__()
        self.cnn0 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3))
        self.cnn1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3))
        self.cnn2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3))
        self.cnn3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3))
        self.cnn4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3))
        self.cnn5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3))
        self.cnn6 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(2, 2))
        self.max_pool22 = nn.MaxPool2d(kernel_size=2)
        self.max_pool12 = nn.MaxPool2d(kernel_size=(1, 2))
        self.norm512 = nn.BatchNorm2d(512, affine=False)
        self.final_layer = nn.Linear(2, 5) # out = 4 units for the bounding box + 1 for the score


    def initialize_weights(self):
        self.attn1.initialize_weights()
        nn.init.xavier_uniform_(self.cnn0.weight)
        nn.init.xavier_uniform_(self.cnn1.weight)
        nn.init.xavier_uniform_(self.cnn2.weight)
        nn.init.xavier_uniform_(self.cnn3.weight)
        nn.init.xavier_uniform_(self.cnn4.weight)
        nn.init.xavier_uniform_(self.cnn5.weight)
        nn.init.xavier_uniform_(self.cnn6.weight)

    def forward(self, x):
        """
        :param x: Tensor of shape (batch, channel, height, width)
        :return: Tensor of shape (batch, score, x1, y1, x2, y2)
        """
        batch_size, _, _, _ = x.shape
        x = nn.functional.relu(self.cnn0(x))
        x = self.max_pool22(nn.functional.relu(self.cnn1(x)))
        x = nn.functional.relu(self.cnn2(x))
        x = self.max_pool12(nn.functional.relu(self.cnn3(x)))
        x = nn.functional.relu(self.cnn4(x))
        x = self.norm512(x)
        x = nn.functional.relu(self.cnn5(x))
        #x = self.norm512(x)
        x = self.max_pool12(x)
        x = nn.functional.relu(self.cnn6(x))
        # Compress vertically
        x = torch.flatten(x, start_dim=2)
        x = self.final_layer(x)

        return x


def get_model(pretrained, eval = False):
    # load the model
    #model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=True)
    model = MyModel()
    # load the model onto the computation device
    if eval:
        model = model.eval()
    else:
        model = model.train()
    return to_best_device(model)


def get_loss(output, expected):
    out_present, out_boundingboxes = output
    expected_present, exp_boundingboxes = expected
    loss_class = F.mse_loss(out_present, expected_present, reduction="sum")
    loss_bb = F.l1_loss(out_boundingboxes, exp_boundingboxes, reduction="none").sum(1)
    loss_bb = loss_bb.sum()
    loss = loss_class + loss_bb
    return loss

