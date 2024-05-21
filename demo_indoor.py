import os
os.chdir("D:\LoFTR-master_modified")
from copy import deepcopy

import torch
import cv2
import numpy as np
import matplotlib.cm as cm
from src.utils.plotting import make_matching_figure
from src.loftr import LoFTR, default_cfg
import time

import cv2
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# The default config uses dual-softmax.
# The outdoor and indoor models share the same config.
# You can change the default values like thr and coarse_match_type.
matcher = LoFTR(config=default_cfg)
matcher.load_state_dict(torch.load("D:\LoFTR-master-scalenet\logs\\tb_logs\\ablation-no-fine\\version_1\checkpoints\epoch=0-auc@5=0.441-auc@10=0.611-auc@20=0.745.ckpt")['state_dict'], strict=True)
matcher = matcher.eval().cuda()
#print(matcher.state_dict())

# In[19]:


default_cfg['coarse']




# Load example images
img0_pth = "assets/phototourism_sample_images/london_bridge_19481797_2295892421.jpg"
img1_pth = "assets/phototourism_sample_images/london_bridge_49190386_5209386933.jpg"
img0_raw = cv2.imread(img0_pth, cv2.IMREAD_GRAYSCALE)
img1_raw = cv2.imread(img1_pth, cv2.IMREAD_GRAYSCALE)
# img0_raw = cv2.resize(img0_raw, (img0_raw.shape[1]//32, img0_raw.shape[0]//32))  # input size shuold be divisible by 8
# img1_raw = cv2.resize(img1_raw, (img1_raw.shape[1]//32, img1_raw.shape[0]//32))
# img0_raw = cv2.resize(img0_raw, (img0_raw.shape[1]//8*8, img0_raw.shape[0]//8*8))  # input size shuold be divisible by 8
# img1_raw = cv2.resize(img1_raw, (img1_raw.shape[1]//8*8, img1_raw.shape[0]//8*8))
img0_raw = cv2.resize(img0_raw, (1024, 1024))
img1_raw = cv2.resize(img1_raw, (1024, 1024))

img0 = torch.from_numpy(img0_raw)[None][None].cuda() / 255.
img1 = torch.from_numpy(img1_raw)[None][None].cuda() / 255.
batch = {'image0': img0, 'image1': img1}


# Inference with LoFTR and get prediction
with torch.no_grad():

    matcher(batch)

    mkpts0 = batch['mkpts0_f'].cpu().numpy()
    mkpts1 = batch['mkpts1_f'].cpu().numpy()
    mconf = batch['mconf'].cpu().numpy()


# In[18]:
# Draw
color = cm.jet(mconf)
text = [
    'Ours',
    'Matches: {}'.format(len(mkpts0)),
]
path = 'D:\\Wnet'
fig = make_matching_figure(img0_raw, img1_raw, mkpts0, mkpts1, color, text=text, dpi=75,  path=path)





