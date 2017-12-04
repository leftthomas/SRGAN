import argparse
import os
from math import log10
from os import listdir

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage
from tqdm import tqdm

import pytorch_ssim
from data_utils import is_image_file
from model import Generator

parser = argparse.ArgumentParser(description='Test Super Resolution')
parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
parser.add_argument('--model_name', default='netG_epoch_4_100.pth', type=str, help='super resolution model name')
opt = parser.parse_args()

UPSCALE_FACTOR = opt.upscale_factor
MODEL_NAME = opt.model_name

results = {'Set5': {'psnr': [], 'ssim': []}, 'Set14': {'psnr': [], 'ssim': []}, 'BSD100': {'psnr': [], 'ssim': []},
           'Urban100': {'psnr': [], 'ssim': []}, 'SunHays80': {'psnr': [], 'ssim': []}}

data_path = 'data/test/SRF_' + str(UPSCALE_FACTOR) + '/data/'
target_path = 'data/test/SRF_' + str(UPSCALE_FACTOR) + '/target/'
images_name = [x for x in listdir(data_path) if is_image_file(x)]

model = Generator(UPSCALE_FACTOR).eval()
if torch.cuda.is_available():
    model = model.cuda()
model.load_state_dict(torch.load('epochs/' + MODEL_NAME))

out_path = 'results/SRF_' + str(UPSCALE_FACTOR) + '/'
if not os.path.exists(out_path):
    os.makedirs(out_path)
for image_name in tqdm(images_name, desc='convert LR images to SR images'):

    image = Image.open(data_path + image_name)
    image = Variable(ToTensor()(image), volatile=True).unsqueeze(0)
    target = Image.open(target_path + image_name)
    target = Variable(ToTensor()(target), volatile=True).unsqueeze(0)
    if torch.cuda.is_available():
        image = image.cuda()
        target = target.cuda()

    out = model(image)
    mse = ((target - out) ** 2).mean()
    psnr = 10 * log10(1 / mse.data.cpu().numpy())
    ssim = pytorch_ssim.ssim(out, target).data.cpu().numpy()
    out_img = ToPILImage()(out[0].data.cpu())
    out_img.save(out_path + 'psnr_%.4f_ssim_%.4f_' % (psnr, ssim) + image_name)
    # save psnr\ssim
    results[image_name.split('_')[0]]['psnr'].append(psnr)
    results[image_name.split('_')[0]]['ssim'].append(psnr)

out_path = 'statistics/SRF_' + str(UPSCALE_FACTOR) + '/'
if not os.path.exists(out_path):
    os.makedirs(out_path)

saved_results = {'psnr': [], 'ssim': []}
for item in results.values():
    psnr = np.array(item['psnr'])
    ssim = np.array(item['ssim'])
    if (len(psnr) == 0) or (len(ssim) == 0):
        psnr = 'No data'
        ssim = 'No data'
    else:
        psnr = psnr.mean()
        ssim = ssim.mean()
    saved_results['psnr'].append(psnr)
    saved_results['ssim'].append(ssim)

data_frame = pd.DataFrame(saved_results, results.keys())
data_frame.to_csv(out_path + 'test_results.csv', index_label='DataSet')
