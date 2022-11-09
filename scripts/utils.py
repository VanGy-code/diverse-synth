import json
import os
import string
import random
import subprocess

import torch
import yaml
from yaml import Loader


# def get_current_lr(epoch, base_lr=0.001, decay_steps=None, decay_rate=None):
#     lr = base_lr
#     if decay_steps is None or decay_rate is None:
#         return lr
#     for i, lr_decay_epoch in enumerate(decay_steps):
#         if epoch >= lr_decay_epoch:
#             lr *= decay_rate[i]
#     return lr


# def adjust_learning_rate(optimizer, epoch, decay_steps, decay_rate):
#     lr = get_current_lr(epoch, decay_rate=decay_rate, decay_steps=decay_steps)
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr


# def adjust_weight_kld(epoch, kld_interval, weight_kld):
#     if epoch < kld_interval:
#         return (epoch / kld_interval) * weight_kld
#     else:
#         return weight_kld

def load_config(config_file):
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=Loader)
    return config


def load_checkpoints(model, optimizer, experiment_directory, device):
    model_files = [
        f for f in os.listdir(experiment_directory)
        if f.startswith("model_")
    ]
    if len(model_files) == 0:
        return 0
    ids = [int(f[6:]) for f in model_files]
    max_id = max(ids)
    model_path = f"{experiment_directory}/model_{max_id}"
    opt_path = f"{experiment_directory}/opt_{max_id}"

    if not (os.path.exists(model_path) and os.path.exists(opt_path)):
        return 0

    print("Loading model checkpoint from {}".format(model_path))
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("Loading optimizer checkpoint from {}".format(opt_path))
    optimizer.load_state_dict(
        torch.load(opt_path, map_location=device)
    )
    return max_id + 1


def save_checkpoints(epoch, model, optimizer, experiment_directory):
    torch.save(
        model.state_dict(),
        f"{experiment_directory}/model_{epoch}"
    )
    torch.save(
        optimizer.state_dict(),
        f"{experiment_directory}/opt_{epoch}"
    )


if __name__ == '__main__':
    print(id_generator(9))
