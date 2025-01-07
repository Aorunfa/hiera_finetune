import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
import hiera
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

# ##  MAE test
# model = hiera.mae_hiera_base_224()
# model.cuda()
# input_size = 224

# transform_list = [
#     transforms.Resize(int((256 / 224) * input_size), interpolation=InterpolationMode.BICUBIC),
#     transforms.CenterCrop(input_size)
# ]

# # The visualization and model need different transforms
# transform_vis  = transforms.Compose(transform_list)
# transform_norm = transforms.Compose(transform_list + [
#     transforms.ToTensor(),
#     transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
# ])

# # Load the image
# img = Image.open("/home/chaofeng/sam2/hiera_git/examples/img/dog.jpg")
# img_vis = transform_vis(img)
# img_norm = transform_norm(img)

# # Get imagenet class as output
# img_norm = img_norm[None, ...].cuda()
# img_norm = torch.tile(img_norm, (2, 1, 1, 1))

# out = model(img_norm)

"""
MAE 训练过程：
    mask生成：
        计算mask连续patch方阵(mask_unit)的后得到的特征图分辨率，基于该分辨率随机mask给定比例的点，保证batch内的每一个图mask的比例相同
    损失计算
        encoder得到没有被mask掉的patch特征
        恢复到原来的patch排列顺序，mask区域填充可学习参数，非mask区域使用encoder得到的特征
        使用vit decoder得到最后的特征图
        标签计算则对原始图片按照最终的下采样stride分块，块状内的channel展平，对齐pred的特征空间。筛选出mask掉的区域
        使用均方误差计算pred和label的差异

Hiera:
    一个有效的图像、视频特征提取器，只使用简单的vit结构。
    结构设计总体思路：
        浅层layer使用高分辨率和小特征维度，深层特征使用低分辨率和大特征维度。
        使用maxpool进行特征图下采样
        前两阶段使用局部注意力机制，后两个阶段使用全局注意力机制
    预训练方式：使用mask-auto-encoder的自监督训练方式，让模型理解图片/视频全局信息
    微调方式：连接一个简单输出头作为decoder
"""


# ## use in1k findtune model to inference in image
# ckpt = '/dev/shm/chaofeng/checkpoint/hiera_base_224.pth'
# model = hiera.hiera_base_224()
# ckpt = torch.load(ckpt)
# model.load_state_dict(ckpt['model_state'])

# model.cuda()

# # Create input transformations
# input_size = 224

# transform_list = [
#     transforms.Resize(int((256 / 224) * input_size), interpolation=InterpolationMode.BICUBIC),
#     transforms.CenterCrop(input_size)
# ]

# # The visualization and model need different transforms
# transform_vis  = transforms.Compose(transform_list)
# transform_norm = transforms.Compose(transform_list + [
#     transforms.ToTensor(),
#     transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
# ])

# # Load the image
# img = Image.open("/home/chaofeng/sam2/hiera_git/examples/img/dog.jpg")
# img_vis = transform_vis(img)
# img_norm = transform_norm(img)

# # Get imagenet class as output
# img_norm = img_norm[None, ...].cuda()
# img_norm = torch.tile(img_norm, (2, 1, 1, 1))

# out = model(img_norm)


# # out = model(img_norm[None, ...].cuda())

# # 207: golden retriever  (imagenet-1k)
# print(out.argmax(dim=-1).item())

####### image #######
## use kinetics-400 findtune model to inference in video
from torchvision.io import read_video
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch, hiera

# I used the following command to sample these videos:
#   ffmpeg -i vid.mp4 -vf "crop=w=min(iw\,ih):h=min(iw\,ih),scale=256:256,fps=fps=30" -ss 00:00:00 -t 00:00:05 output.mp4
vid_path = "/home/chaofeng/sam2/hiera_git/examples/vid/dog.mp4"  # Also try "goat.mp4"!

# Load the frames
frames, audio, info = read_video(vid_path, pts_unit='sec', output_format='THWC')
frames = frames.float() / 255  # Convert from byte to float
print(info['video_fps'])  # Should be 30

frames = torch.stack([frames[:64], frames[64:128]], dim=0)
frames = frames[:, ::4]  # Sample every 4 frames
print(frames.shape)

# Interpolate the frames to 224 and put channels first
frames = frames.permute(0, 4, 1, 2, 3).contiguous()
frames = F.interpolate(frames, size=(16, 224, 224), mode="trilinear")
print(frames.shape)

# Normalize the clip
frames = frames - torch.tensor([0.45, 0.45, 0.45]).view(1, -1, 1, 1, 1)     # Subtract mean
frames = frames / torch.tensor([0.225, 0.225, 0.255]).view(1, -1, 1, 1, 1)  # Divide by std

# model
model = hiera.hiera_base_16x224()
ckpt = '/dev/shm/chaofeng/checkpoint/hiera_base_16x224.pth'
ckpt = torch.load(ckpt)
model.load_state_dict(ckpt['model_state'])
model.cuda()

# Get kinetics classes as output
out = model(frames.cuda())

# Average results over the clips
out = out.mean(0)

# 363: training dog  (kinetics-400)
# 125: feeding goat  (kinetics-400)
print(out.argmax(dim=-1).item())

"""
视频模型结构设计
    pos embed在图片的spatial embed基础上增加temporal embed
"""
