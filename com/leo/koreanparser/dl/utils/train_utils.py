import os
import sys
import time
import numpy as np
from typing import Union

import torch
from torch.nn import Module

from com.leo.koreanparser.dl.model import get_model
from com.leo.koreanparser.dl.utils.tensor_helper import to_best_device

def update_optimizer(optimizer, lr):
    for i, param_group in enumerate(optimizer.param_groups):
        param_group["lr"] = lr


def train_epocs(model, optimizer, train_dl, val_dl, models_rep, loss_computer, epochs=10, threshold: float=0.5, scheduler=None):
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
            optimizer.zero_grad()
            batch = y_class.shape[0]
            x = to_best_device(x).float()
            y_class = to_best_device(y_class).float()
            y_bb = to_best_device(y_bb).float()
            out_class, out_bb = model(x)
            loss = loss_computer.loss(out_class, y_class, out_bb, y_bb)
            loss.backward()
            optimizer.step()
            idx += 1
            total += batch
            sum_loss += loss.item()
        scheduler.step()
        losses.append(sum_loss)
        if total > 0:
            train_loss = sum_loss/total
        val_loss, val_acc, val_box_acc = val_metrics(model, val_dl, loss_computer, threshold=threshold)
        str_box_loss_formatted = " - ".join(["%.3f" % val for val in val_box_acc])
        print("train_loss %.3f val_loss %.3f val_acc %.3f ---- Loss Boxes : %s" % (train_loss, val_loss, val_acc, str_box_loss_formatted))
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


def val_metrics(model, valid_dl, loss_computer, threshold: float=0.5):
    model.eval()
    total = 0
    sum_loss = 0
    correct = 0
    val_losses = []
    for x, y_class, y_bb in valid_dl:
        batch = y_class.shape[0]
        x = to_best_device(x).float()
        y_class = to_best_device(y_class).float()
        y_bb = to_best_device(y_bb).float()
        out_class, out_bb = model(x)
        losses = loss_computer.losses(out_class, y_class, out_bb, y_bb)
        val_losses.append([lo.item() for lo in losses])
        loss = loss_computer.aggregate_losses(losses)
        subbed_hat = out_class >= threshold
        subbed = y_class >= threshold
        correct += subbed_hat.squeeze().eq(subbed).sum().item()
        sum_loss += loss.item()
        total += batch
    appended = np.sum(val_losses, axis=0)

    return sum_loss/total, correct/total, appended / total


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

def do_lod_specific_model(model_path: str, model: Module) -> Module:
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    return model

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