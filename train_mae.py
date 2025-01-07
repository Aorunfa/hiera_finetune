import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from hiera import hiera
from tools.utils import save_metric, save_checkpoint, print_loss, strict_load
from tools.dataset import MaeData
import warnings
import time

class Config(object):
    def __init__(self):
        self.epochs = 300
        self.optimizer = AdamW
        self.lr = 1e-4
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
        self.metric_csv = './record/metric_mae.csv'

        self.train_dir = '/dev/shm/chaofeng/imagenet/root/train'
        self.val_dir = '/dev/shm/chaofeng/imagenet/root/test'
        self.batch_size = 32
        self.num_worker = 8

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

    # do train
    for epoch in range(args.epochs):
        for step, batch in enumerate(train_loader):
            step += 1 + epoch * len(train_loader)
            
            if step < args.warmup:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")            
                    scheduler_warmup.step()
            else:
                scheduler.step()
                
            with autocast(args.device, enabled=args.amp, dtype=torch.bfloat16):
                output = model(batch.to(device))
                loss = output[0]
                loss_item = loss.data.item()

            if args.amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            if step % args.accumulate == 0:
                if args.amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                    optimizer.step()
                optimizer.zero_grad()
                
                current_lr = optimizer.param_groups[0]['lr']
                print_loss(step, loss_item, current_lr)
            
            if step % args.val_iter == 0:
                val_loss = val(args, model, val_loader)
                metric = {
                    'time': time.strftime("%m-%d %H:%M"),
                    'step': step,
                    'loss_train': loss_item,
                    'lr': current_lr,
                    'loss_val': val_loss,
                }
                save_metric(metric, args.metric_csv)
            
            # if step % args.save_iter == 0:
                # save_checkpoint(model, step, args.save_dir)

@torch.no_grad()          
def val(args, model, val_loader):
    # use val loss for metric
    model.eval()
    loss_tol = 0.
    for batch in val_loader:
        output = model(batch.to(args.device))
        loss = output[0]
        loss_tol += loss.data.item()
    model.train()
    val_loss = loss_tol / len(val_loader)
    print('val loss ------', val_loss)
    return val_loss   

if __name__ == '__main__':
    args = Config()
    train_loader = DataLoader(MaeData(args.train_dir), batch_size=args.batch_size, num_workers=args.num_worker)
    val_loader = DataLoader(MaeData(args.val_dir), batch_size=args.batch_size, num_workers=args.num_worker)
    model = hiera.mae_hiera_base_224()

    if args.resum_path != '':
        ckpt = torch.load(args.resum_path)
        ckpt, unload = strict_load(model.state_dict(), ckpt['model_state'])
        model.load_state_dict(ckpt)

    train(args, model, train_loader, val_loader)


    """
    nohup /var/lib/anaconda3/envs/sam2/bin/python /home/chaofeng/sam2/hiera_git/train_mae.py > /home/chaofeng/sam2/hiera_git/n_mae.log 2>&1 &
    """

    """
    注意： mae 当前训练不稳定， 需要回到原始git repo查找原因
    """

