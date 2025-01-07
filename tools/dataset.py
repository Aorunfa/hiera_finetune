import os
from PIL import Image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import Dataset
import pandas as pd
import torch


def int2str_filled(i):
    width = 6
    return '_{:0{}d}'.format(i, width)

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