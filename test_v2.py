import numpy as np
import os
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch
import torch.nn.functional as F
import utils
import cv2
from natsort import natsorted
from glob import glob
from basicsr.models.archs.frac_fftformer_arch import frac_fftformer
from basicsr.models.archs.fftformer_arch import fftformer

from skimage import img_as_ubyte
#%%
def best_fit(X,Y):
    xbar = sum(X)/len(X)
    ybar = sum(Y)/len(Y)
    n = len(Y)
    numer = sum([xi*yi for xi, yi in zip(X,Y)]) - n * xbar * ybar
    denom = sum([xi**2 for xi in X]) - n * xbar**2
    b = numer/denom
    # if denom == 0:
    #     b=1
    a = ybar - b*xbar
    return a,b

#%%

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
parser = argparse.ArgumentParser(description='Single Image Motion Deblurring using Restormer')

parser.add_argument('--input_dir', default='RainDrop/', type=str, help='Directory of test images')
parser.add_argument('--result_dir', default='./results/', type=str, help='Directory for results')
parser.add_argument('--blur_weights', default='checkpoints/net_g_blur_removal.pth', type=str, help='Path to blur weights')
parser.add_argument('--drop_weights', default='checkpoints/net_g_drop_removal.pth', type=str, help='Path to drop weights')
parser.add_argument('--dataset', default='ReainDrop_removal', type=str, help='Test Dataset') 
args = parser.parse_args()

####### Load drop yaml #######
drop_yaml_file = 'options/test/raindrop_raindrop_removal.yml'
import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

x = yaml.load(open(drop_yaml_file, mode='r'), Loader=Loader)

s = x['network_g'].pop('type')
##########################

model_drop = fftformer(**x['network_g'])

checkpoint = torch.load(args.drop_weights)
model_drop.load_state_dict(checkpoint['params'])
print("===>Testing drop removal using weights: ",args.drop_weights)
model_drop.cuda()
model_drop = nn.DataParallel(model_drop)
model_drop.eval()


####### Load blur yaml #######
blur_yaml_file = 'options/test/raindrop_blur_removal.yml'
import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

x = yaml.load(open(blur_yaml_file, mode='r'), Loader=Loader)

s = x['network_g'].pop('type')
##########################

model_blur = frac_fftformer(**x['network_g'])

checkpoint = torch.load(args.blur_weights)
model_blur.load_state_dict(checkpoint['params'])
print("===>Testing blur removal using weights: ",args.blur_weights)
model_blur.cuda()
model_blur = nn.DataParallel(model_blur)
model_blur.eval()


factor = 32
dataset = args.dataset
result_dir  = os.path.join(args.result_dir, dataset)
os.makedirs(result_dir, exist_ok=True)
inp_dir = args.input_dir #.path.join(args.input_dir, 'test', dataset, 'input')

files = sorted(glob(os.path.join(inp_dir, '/*.png'))) 


print('No. of files to be tested:',len(files))
with torch.no_grad():
    for i in tqdm(range(len(files))):
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()

        img_data = np.float32(utils.load_img(files[i]))/255.0
        img = torch.from_numpy(img_data).permute(2,0,1)
        input_ = img.unsqueeze(0).cuda()
        
        h,w = input_.shape[2], input_.shape[3]
        H,W = ((h+factor)//factor)*factor, ((w+factor)//factor)*factor
        padh = H-h if h%factor!=0 else 0
        padw = W-w if w%factor!=0 else 0
        input_ = F.pad(input_, (0,padw,0,padh), 'reflect')
        
        input_data = input_ 
        drop_removed = model_drop(input_data)
        drop_removed = torch.clamp(drop_removed,0,1)
        
        blur_removed = model_blur(drop_removed)[0]
        blur_removed = blur_removed[:,:,:h,:w]
        
        drop_removed = drop_removed[:,:,:h,:w].cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
        ref_data = np.float32(drop_removed*255.0)

        restored = torch.clamp(blur_removed,0,1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
        restored = np.float32(restored*255.0)
        
        img_data = np.float32(np.zeros_like(img_data))
        
        c1, m1 = best_fit(ref_data[:,:,0].ravel(), restored[:,:,0].ravel())
        img_data[:,:,0] = (restored[:,:,0] - c1)/m1
        
        c2, m2 = best_fit(ref_data[:,:,1].ravel(), restored[:,:,1].ravel())
        img_data[:,:,1] = (restored[:,:,1] - c2)/m2
        
        c3, m3 = best_fit(ref_data[:,:,2].ravel(), restored[:,:,2].ravel())
        img_data[:,:,2] = (restored[:,:,2] - c3)/m3
        
        img_data[img_data<0] = 0
        img_data[img_data>255] = 255
        
        filename = os.path.basename(files[i])
        utils.save_img(result_dir + '/' + filename, np.uint8(img_data))
        
        






