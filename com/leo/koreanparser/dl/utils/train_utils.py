import os
import sys
import time
from typing import Union

import torch
from torch.nn import Module

from com.leo.koreanparser.dl.model import get_model
from com.leo.koreanparser.dl.utils.tensor_helper import to_best_device
import torch.nn.functional as F

def update_optimizer(optimizer, lr):
    for i, param_group in enumerate(optimizer.param_groups):
        param_group["lr"] = lr


def train_epocs(model, optimizer, train_dl, val_dl, models_rep, epochs=10):
    best_model = to_best_device(get_model())
    start = time.time()
    losses = []
    min_loss = sys.maxsize
    do_save = True
    idx = 0
    for i in range(epochs):
        model.train()
        total = 0
        sum_loss = 0
        for x, y_class, y_bb in train_dl:
            batch = y_class.shape[0]
            x = to_best_device(x).float()
            y_class = to_best_device(y_class).float()
            y_bb = to_best_device(y_bb).float()
            out_class, out_bb = model(x)
            loss_class = F.binary_cross_entropy(out_class, y_class.unsqueeze(1), reduction="sum")
            loss_bb = F.l1_loss(out_bb, y_bb, reduction="none").sum(1)
            loss_bb = loss_bb.sum()
            loss = loss_class + loss_bb / 4
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            idx += 1
            total += batch
            sum_loss += loss.item()
        losses.append(sum_loss)
        train_loss = sum_loss/total
        val_loss, val_acc = val_metrics(model, val_dl)
        print("train_loss %.3f val_loss %.3f val_acc %.3f" % (train_loss, val_loss, val_acc))
        if sum_loss < min_loss:
            do_save = True
            best_model.load_state_dict(model.state_dict())
            min_loss = sum_loss
        else:
            if do_save:
                print(f'[{idx}] Best loss so far is {min_loss} so we will save in best')
                torch.save(best_model.state_dict(), f"{models_rep}/best.pt")
            do_save = False
    end = time.time()
    print(f"It took {end - start}")
    if do_save:
        print(f'[END] Best loss was {min_loss} so we will save in best')
        torch.save(best_model.state_dict(), f"{models_rep}/best.pt")
    return sum_loss/total


def val_metrics(model, valid_dl):
    model.eval()
    total = 0
    sum_loss = 0
    correct = 0
    for x, y_class, y_bb in valid_dl:
        batch = y_class.shape[0]
        x = x.cuda().float()
        y_class = y_class.cuda().float()
        y_bb = y_bb.cuda().float()
        out_class, out_bb = model(x)
        loss_class = F.binary_cross_entropy(out_class, y_class.unsqueeze(1), reduction="sum")
        loss_bb = F.l1_loss(out_bb, y_bb, reduction="none").sum(1)
        loss_bb = loss_bb.sum()
        loss = loss_class + loss_bb / 4
        _, pred = torch.max(out_class, 1)
        correct += pred.eq(y_class).sum().item()
        sum_loss += loss.item()
        total += batch
    return sum_loss/total, correct/total


def get_last_model_params(models_rep) -> Union[str, None]:
    if not os.path.exists(models_rep):
        os.makedirs(models_rep)
    else:
        file_list = os.listdir(models_rep)
        file_list = [f for f in file_list if f[-3:] == '.pt']
        file_list.sort(reverse=True)
        if len(file_list) > 0:
            return f"{models_rep}/{file_list[0]}"
    return None


def do_load_model(models_rep: str, model: Module, exit_on_error: bool = False) -> bool:
    last_model_file = get_last_model_params(models_rep)
    if not last_model_file:
        print(f"No parameters to load from {models_rep}")
        if exit_on_error:
            exit(-1)
        else:
            return False
    print(f"Loading parameters from {last_model_file}")
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(last_model_file))
    else:
        model.load_state_dict(torch.load(last_model_file, map_location=torch.device('cpu')))
    return True