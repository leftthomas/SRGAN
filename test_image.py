import argparse

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage

from model import Generator

parser = argparse.ArgumentParser(description='Test Super Resolution')
parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
parser.add_argument('--image_name', type=str, help='low resolution image name')
parser.add_argument('--model_name', default='netG_epoch_4_100.pth', type=str, help='super resolution model name')
opt = parser.parse_args()

UPSCALE_FACTOR = opt.upscale_factor
IMAGE_NAME = opt.image_name
MODEL_NAME = opt.model_name

model = Generator(UPSCALE_FACTOR).eval()
if torch.cuda.is_available():
    model = model.cuda()
# for cpu
# model.load_state_dict(torch.load('epochs/' + MODEL_NAME, map_location=lambda storage, loc: storage))
model.load_state_dict(torch.load('epochs/' + MODEL_NAME))

image = Image.open(IMAGE_NAME)
image = Variable(ToTensor()(image), volatile=True).unsqueeze(0)
if torch.cuda.is_available():
    image = image.cuda()

out = model(image)
out_img = ToPILImage()(out[0].data.cpu())
out_img.save('out_' + IMAGE_NAME)
