"""
以tiny-k-400数据集为例进行微调
下载tiny数据集：https://github.com/Tramac/tiny-kinetics-400
"""
import os

import torch.amp
import torch.utils
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
from hiera import hiera
# import hiera
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
import pandas as pd
import numpy as np
import torch.nn.functional as F
from PIL import Image
from sklearn import metrics


def print_loss(step_num, loss, current_lr):
    l = loss.data.item()
    print("step: {:0>8d}{:>8s} loss: {:.4f} lr: {:.8f}".format(step_num, '', l, current_lr))

def int2str_filled(i):
    width = 6
    return '_{:0{}d}'.format(i, width)

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
    # print(unload)
    return model_state, unload

class TinyData(Dataset):
    def __init__(self, csv_path, save_dir, start_idx=10):
        self.video_path, self.labels, self.labels_idx = self.get_img_data(csv_path, save_dir)        
        self.transform_norm = transforms.Compose([
                                                transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
                                            ])
        self.start_idx = start_idx
    
    def get_img_data(self, csv_path, save_dir):
        df = pd.read_csv(csv_path)
        df = df.sample(frac=1, random_state=123)
        df['video_frames_path'] = df.apply(lambda x: os.path.join(save_dir, 
                                                                x['label'].replace(' ', '_'), 
                                                                x['youtube_id'] +
                                                                int2str_filled(x['time_start']) +
                                                                int2str_filled(x['time_end'])), axis=1)


        video_path = df['video_frames_path'].to_list()
        labels = df['label'].to_list()
        labels_idx = sorted(df['label'].unique().tolist())
        labels_idx = dict(zip(labels_idx, [i for i in range(len(labels_idx))]))
        labels = [labels_idx[l] for l in labels]
        return video_path, labels, labels_idx

    def __len__(self, ):
        return len(self.video_path)
    
    def read_images(self, imgs_path):
        # Load the frames
        imgs = []
        for img_path in imgs_path:
            img = Image.open(img_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            imgs.append(self.transform_norm(img))

        return torch.stack(imgs, dim=0)
        
    def __getitem__(self, index):
        frame_skip = 4
        frame_nums = 16
        video_path = self.video_path[index]
        label = self.labels[index]
        images = os.listdir(video_path)
        images = sorted(images, key=lambda x: int(x.replace('frame_', '').replace('.jpg', '')))
        
        # random sample continous
        # start_idx = np.random.randint(0, len(images) - frame_nums * frame_skip)
        start_idx = self.start_idx
        images = images[start_idx: start_idx + frame_nums * frame_skip : frame_skip]
        images = [os.path.join(video_path, f) for f in images]
        images = self.read_images(images)
        return {'images': images, 'label': label}
    
def collect_fun(batch):
    images = torch.stack([b['images'] for b in batch], dim=0)
    labels = torch.Tensor([b['label'] for b in batch]).to(torch.int64)
    images = images.permute(0, 2, 1, 3, 4).contiguous()
    return images, labels

class Config(object):
    def __init__(self):
        self.epochs = 100
        self.lr = 1e-6
        self.final_lr = 0.001 # rate
        self.optimizer = AdamW
        self.data_dir = '/dev/shm/chaofeng/tiny_k400/tiny-kinetics-400/data_30fps_frames/'
        self.device = 'cuda'
        self.warmup = 500
        self.accumulate = 10
        self.val_iter = 100 * self.accumulate
        self.grad_clip = 1.
        self.amp = False
        # self.train_csv = '/dev/shm/chaofeng/tiny_k400/tiny-kinetics-400/annotations/tiny_train.csv'
        # self.val_csv = '/dev/shm/chaofeng/tiny_k400/tiny-kinetics-400/annotations/tiny_val.csv'

        self.train_csv = '/dev/shm/chaofeng/tiny_k400/tiny-kinetics-400/annotations/tiny_train.csv'
        self.val_csv = '/dev/shm/chaofeng/tiny_k400/tiny-kinetics-400/annotations/tiny_train.csv'


        self.batchsize = 8
        self.numworker = 4

        self.save_iter = self.val_iter

def train(config: Config, model:torch.nn.Module, train_loader, val_loader):
    # model
    model.to(config.device)
    model.train()

    # optimizer
    optimizer = config.optimizer(model.parameters(), lr=config.lr, weight_decay=0.01)
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=config.warmup)
    opt_scheduler = LinearLR(optimizer, start_factor=1, end_factor=config.final_lr, total_iters=config.epochs * len(train_loader))

    # scaler
    if config.amp:
        scaler = GradScaler(config.device)

    # do train
    for epoch in range(config.epochs):
        for step, (images, labels) in enumerate(train_loader):
            step += 1 + epoch * len(train_loader)
            
            if step < config.warmup:
                warmup_scheduler.step()
            else:
                opt_scheduler.step()
                
            with autocast(config.device, enabled=config.amp, dtype=torch.bfloat16):
                output = model(images.to(config.device))
                loss = F.cross_entropy(output, labels.to(config.device), reduction='mean')

                if config.amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                if step % config.accumulate == 0:
                    if config.amp:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                        optimizer.step()
                    optimizer.zero_grad()

                    current_lr = optimizer.param_groups[0]['lr']
                    print_loss(step, loss, current_lr)
                
                if step % config.val_iter == 0:
                    # vlidate
                    val(config, model, val_loader)
                
                if step % config.save_iter == 0:
                    # save
                    pass

@torch.no_grad()          
def val(config, model, val_loader):
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)

    model.eval()
    for images, labels in val_loader:
        output = model(images.to(config.device))
        preds = output.argmax(-1).cpu().numpy()
        labels = labels.cpu().numpy()
        labels_all = np.append(labels_all, labels)
        predict_all = np.append(predict_all, preds)
    acc = metrics.accuracy_score(labels_all, predict_all)
    model.train()
    print('------- acc ---------', acc)
    return acc
if __name__ == '__main__':
    config = Config()
    train_dataset = TinyData(config.train_csv, config.data_dir, start_idx=10)
    val_dataset = TinyData(config.val_csv, config.data_dir, start_idx=20)
    train_loader = DataLoader(train_dataset, batch_size=config.batchsize, num_workers=config.numworker, collate_fn=collect_fun)
    val_loader = DataLoader(val_dataset, batch_size=config.batchsize * 2, num_workers=config.numworker, collate_fn=collect_fun)

    # load mae pretrain and frezze it
    model = hiera.hiera_base_16x224()
    ckpt = torch.load('/dev/shm/chaofeng/checkpoint/mae_hiera_base_16x224.pth')
    ckpt, unload = strict_load(model.state_dict(), ckpt['model_state'])
    model.load_state_dict(ckpt)

    # for k, p in model.named_parameters(): # freez output head
    #     if k in unload:
    #         print(k)
    #         p.requires_grad = True
    #     else:
    #         p.requires_grad = False

    train(config, model, train_loader, val_loader)


    """
    nohup /var/lib/anaconda3/envs/sam2/bin/python /home/chaofeng/hiera_finetune/train_ft.py > /home/chaofeng/hiera_finetune/n_scrach.log 2>&1 &
    """
    """
    需要扩大数据量
    """

