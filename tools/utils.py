import os
import pandas as pd
import torch
import time


def save_metric(data:dict, save_csv):
    with open(save_csv, 'a') as file:
        if isinstance(data, dict):
            data = pd.DataFrame(data, index=[0])
        data.to_csv(file,
                    sep='\t',
                    index=False,
                    header=not os.stat(save_csv).st_size > 0)

def save_checkpoint(model, step, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    suffix = '%s-%s.pth' % (step, time.strftime("%Y-%H-%M-%S"))
    model.eval()
    torch.save(model.state_dict(), os.path.join(save_dir, suffix))
    model.train()

def print_loss(step_num, loss_item, current_lr):
    print("step: {:0>8d}{:>8s} loss: {:.4f} lr: {:.8f}".format(step_num, '', loss_item, current_lr))

def strict_load(model_state, ckpt_state):
    unload, load = [], []
    ckpt_keys = ckpt_state.keys()
    for k, v in model_state.items():
        if k in ckpt_keys and ckpt_state[k].shape == v.shape:
            model_state[k] = ckpt_state[k]
            load.append(k)
        else:
            unload.append(k)
    print('load total: %d/%d' % (len(load), len(load) + len(unload)))
    return model_state, unload