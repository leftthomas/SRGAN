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

results_Set5_psnr = []
results_Set5_ssim = []
results_Set14_psnr = []
results_Set14_ssim = []
results_BSD100_psnr = []
results_BSD100_ssim = []
results_Urban100_psnr = []
results_Urban100_ssim = []
results_SunHays80_psnr = []
results_SunHays80_ssim = []

data_path = 'data/test/SRF_' + str(UPSCALE_FACTOR) + '/data/'
target_path = 'data/test/SRF_' + str(UPSCALE_FACTOR) + '/target/'
images_name = [x for x in listdir(data_path) if is_image_file(x)]

model = Generator(UPSCALE_FACTOR)
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
    if image_name.startswith('Set5'):
        results_Set5_psnr.append(psnr)
        results_Set5_ssim.append(ssim)
    elif image_name.startswith('Set14'):
        results_Set14_psnr.append(psnr)
        results_Set14_ssim.append(ssim)
    elif image_name.startswith('BSD100'):
        results_BSD100_psnr.append(psnr)
        results_BSD100_ssim.append(ssim)
    elif image_name.startswith('Urban100'):
        results_Urban100_psnr.append(psnr)
        results_Urban100_ssim.append(ssim)
    elif image_name.startswith('SunHays80'):
        results_SunHays80_psnr.append(psnr)
        results_SunHays80_ssim.append(ssim)

out_path = 'statistics/SRF_' + str(UPSCALE_FACTOR) + '/'
if not os.path.exists(out_path):
    os.makedirs(out_path)
data = {}
index = []
psnr = []
ssim = []
if len(results_Set5_psnr) > 0 and len(results_Set5_ssim) > 0:
    psnr.append(np.array(results_Set5_psnr).mean())
    ssim.append(np.array(results_Set5_ssim).mean())
    index.append('Set5')
if len(results_Set14_psnr) > 0 and len(results_Set14_ssim) > 0:
    psnr.append(np.array(results_Set14_psnr).mean())
    ssim.append(np.array(results_Set14_ssim).mean())
    index.append('Set14')
if len(results_BSD100_psnr) > 0 and len(results_BSD100_ssim) > 0:
    psnr.append(np.array(results_BSD100_psnr).mean())
    ssim.append(np.array(results_BSD100_ssim).mean())
    index.append('BSD100')
if len(results_Urban100_psnr) > 0 and len(results_Urban100_ssim) > 0:
    psnr.append(np.array(results_Urban100_psnr).mean())
    ssim.append(np.array(results_Urban100_ssim).mean())
    index.append('Urban100')
if len(results_SunHays80_psnr) > 0 and len(results_SunHays80_ssim) > 0:
    psnr.append(np.array(results_SunHays80_psnr).mean())
    ssim.append(np.array(results_SunHays80_ssim).mean())
    index.append('SunHays80')
data['PSNR'] = psnr
data['SSIM'] = ssim
data_frame = pd.DataFrame(data, index)
data_frame.to_csv(out_path + 'test_results.csv', index_label='DataSet')
