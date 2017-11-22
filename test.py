import argparse
import os
from math import log10
from os import listdir

import pytorch_ssim
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage
from tqdm import tqdm

from data_utils import is_image_file
from model import Generator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test Super Resolution')
    parser.add_argument('--upscale_factor', default=3, type=int, help='super resolution upscale factor')
    parser.add_argument('--model_name', default='netG_epoch_3_100.pth', type=str, help='super resolution model name')
    opt = parser.parse_args()

    UPSCALE_FACTOR = opt.upscale_factor
    MODEL_NAME = opt.model_name

    data_path = 'data/test/SRF_' + str(UPSCALE_FACTOR) + '/data/'
    target_path = 'data/test/SRF_' + str(UPSCALE_FACTOR) + '/target/'
    images_name = [x for x in listdir(data_path) if is_image_file(x)]

    model = Generator(upscale_factor=UPSCALE_FACTOR)
    if torch.cuda.is_available():
        model = model.cuda()
    model.load_state_dict(torch.load('epochs/' + MODEL_NAME))

    out_path = 'results/SRF_' + str(UPSCALE_FACTOR) + '/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    for image_name in tqdm(images_name, desc='convert LR images to SR images'):

        image = Image.open(data_path + image_name)
        image = Variable(ToTensor()(image))
        target = Image.open(target_path + image_name)
        target = Variable(ToTensor()(target))
        if torch.cuda.is_available():
            image = image.cuda()
            target = target.cuda()

        out = model(image.unsqueeze(0))[0]
        mse = ((target - out) ** 2).mean()
        psnr = 10 * log10(1 / mse.data.cpu().numpy())
        ssim = pytorch_ssim.ssim(out, target)
        out_img = ToPILImage()(out.data)
        out_img.save(out_path + 'psnr_%.4f_ssim_%.4f_' % (psnr, ssim) + image_name)
