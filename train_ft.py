import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
import numpy as np
import torch.nn.functional as F
from sklearn import metrics
import warnings
from hiera import hiera
from tools.dataset import TinyData, collect_fun
from tools.utils import save_metric, save_checkpoint, print_loss, strict_load
import time

class Config(object):
    def __init__(self):
        self.epochs = 300
        self.optimizer = AdamW
        self.lr = 1e-3
        self.final_lr = 0.001 # rate
        self.grad_clip = 1.

        self.amp = False
        self.device = 'cuda'
        self.warmup = 250
        self.warmup_start_frac = 0.01
        self.accumulate = 10
        self.val_iter = 100 * self.accumulate
        self.save_iter = self.val_iter
        self.save_dir = './record/checkpoint'
        self.metric_csv = './record/metric.csv'

        self.train_csv = '/dev/shm/chaofeng/tiny_k400/tiny-kinetics-400/annotations/tiny_train.csv'
        self.val_csv = '/dev/shm/chaofeng/tiny_k400/tiny-kinetics-400/annotations/tiny_train.csv'
        self.data_dir = '/dev/shm/chaofeng/tiny_k400/tiny-kinetics-400/data_30fps_frames/'
        self.batch_size = 8
        self.num_worker = 4

        self.resum_path = '' # '/dev/shm/chaofeng/checkpoint/mae_hiera_base_16x224.pth'

def train(args: Config, model:torch.nn.Module, train_loader, val_loader):
    # model
    device = args.device
    model.to(device)
    model.train()

    # optimizer
    optimizer = args.optimizer(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler_warmup = LinearLR(optimizer, start_factor=args.warmup_start_frac, total_iters=args.warmup)
    scheduler = LinearLR(optimizer, start_factor=1, end_factor=args.final_lr, total_iters=args.epochs * len(train_loader))

    # scaler
    if args.amp:
        scaler = GradScaler(device)

    val(args, model, val_loader)
    # do train
    for epoch in range(args.epochs):
        for step, (images, labels) in enumerate(train_loader):
            step += 1 + epoch * len(train_loader)
            
            if step < args.warmup:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")            
                    scheduler_warmup.step()
            else:
                scheduler.step()
                
            with autocast(device, enabled=args.amp, dtype=torch.bfloat16):
                output = model(images.to(device))
                loss = F.cross_entropy(output, labels.to(device), reduction='mean')
                loss_item = loss.data.item()

            if args.amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            if step % args.accumulate == 0:
                if args.amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                    optimizer.step()
                optimizer.zero_grad()

                current_lr = optimizer.param_groups[0]['lr']
                print_loss(step, loss_item, current_lr)
            
            if step % args.val_iter == 0:
                acc = val(args, model, val_loader)
                metric = {
                    'time': time.strftime("%m-%d %H:%M"),
                    'step': step,
                    'loss': loss_item,
                    'lr': current_lr,
                    'acc': acc,
                }
                save_metric(metric, args.metric_csv)
            
            # if step % args.save_iter == 0:
                # save_checkpoint(model, step, args.save_dir)

@torch.no_grad()          
def val(args, model, val_loader):
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)

    model.eval()
    for images, labels in val_loader:
        output = model(images.to(args.device))
        preds = output.argmax(-1).cpu().numpy()
        labels = labels.cpu().numpy()
        labels_all = np.append(labels_all, labels)
        predict_all = np.append(predict_all, preds)
    acc = metrics.accuracy_score(labels_all, predict_all)
    model.train()
    print('------- acc ---------', acc)
    return acc

if __name__ == '__main__':
    args = Config()
    train_dataset = TinyData(args.train_csv, args.data_dir, start_idx=10)
    val_dataset = TinyData(args.val_csv, args.data_dir, start_idx=40)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_worker, collate_fn=collect_fun)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size * 2, num_workers=args.num_worker, collate_fn=collect_fun)

    model = hiera.hiera_base_16x224()
    if args.resum_path != '':
        ckpt = torch.load(args.resum_path)
        ckpt, unload = strict_load(model.state_dict(), ckpt['model_state'])
        model.load_state_dict(ckpt)
        
        # frezze encoder decoder
        for k, p in model.named_parameters(): # freez output head
            if k in unload:
                print(k)
                p.requires_grad = True
            else:
                p.requires_grad = False

    train(args, model, train_loader, val_loader)

    """
    nohup /var/lib/anaconda3/envs/sam2/bin/python /home/chaofeng/hiera_finetune/train_ft.py > /home/chaofeng/hiera_finetune/n_scrach.log 2>&1 &
    """

