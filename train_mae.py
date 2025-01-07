import os

import torch.amp
import torch.utils
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import torch
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
import hiera
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.amp.autocast_mode import autocast

def print_loss(step_num, loss):
    l = loss.data.item()
    print("step: {:0>8d}{:>8s} loss: {:.4f}".format(step_num, '', l))

class MaeData(Dataset):
    def __init__(self, image_dir):
        self.image_path = self.get_img_path(image_dir, recursive=True)
        
        #### for test
        import numpy as np
        np.random.shuffle(self.image_path)
        self.image_path = self.image_path[:2500]
        #####
        
        input_size = 224
        self.transform_norm = transforms.Compose([
                                                transforms.Resize(int((256 / 224) * input_size), interpolation=InterpolationMode.BICUBIC),
                                                transforms.CenterCrop(input_size),
                                                transforms.ToTensor(),
                                                transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
                                            ])
    
    def get_img_path(self, image_dir, recursive=False):
        if not recursive:
            file_list = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith(('jpg', 'png'))]
        else:
            file_list = []
            for dirpath, dirnames, filenames in os.walk(image_dir):
                for filename in filenames:
                    if filename.lower().endswith(('jpeg', 'jpg', 'png')):
                        file_path = os.path.join(dirpath, filename)
                        file_list.append(file_path)
        return file_list

    def __len__(self, ):
        return len(self.image_path)
    
    def __getitem__(self, index):
        im_path = self.image_path[index]
        img = Image.open(im_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img_norm = self.transform_norm(img)
        return img_norm

class Config(object):
    def __init__(self):
        self.epochs = 50
        self.lr = 0.1
        self.final_lr = 0.001 # rate
        self.optimizer = AdamW
        self.data_dir = '/dev/shm/chaofeng/imagenet/root/train'
        self.device = 'cuda'
        self.warmup = 1000
        self.accumulate = 10
        self.grad_clip = 1.
        self.amp = False


def train(config: Config, model:torch.nn.Module, train_loader):
    # model
    model.to(config.device)
    model.train()
    
    # optimizer
    optimizer = config.optimizer(model.parameters(), lr=config.lr)
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=config.warmup)
    opt_scheduler = LinearLR(optimizer, start_factor=1, end_factor=config.final_lr, total_iters=config.epochs * len(train_loader))

    # scaler
    if config.amp:
        scaler = torch.amp.grad_scaler.GradScaler(config.device)

    # do train
    for epoch in range(config.epochs):
        for step, batch in enumerate(train_loader):
            step += epoch * len(train_loader)
            
            if step < config.warmup:
                warmup_scheduler.step()
            else:
                opt_scheduler.step()
                
            with autocast(config.device, enabled=config.amp):
                output = model(batch.to(config.device))
                loss = output[0]
                optimizer.zero_grad()

                if config.amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                if step % config.accumulate == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip)
                    if config.amp:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    print_loss(step, loss)

if __name__ == '__main__':
    config = Config()
    dataset = MaeData(config.data_dir)
    train_loader = DataLoader(dataset, batch_size=32, num_workers=16)
    model = hiera.mae_hiera_base_224()
    train(config, model, train_loader)
    """
    nohup /var/lib/anaconda3/envs/sam2/bin/python /home/chaofeng/sam2/hiera_git/train_mae.py > /home/chaofeng/sam2/hiera_git/n_mae.log 2>&1 &
    """

    """
    注意： mae 当前训练不稳定， 需要回到原始git repo查找原因
    """

