import argparse

import numpy as np
import skvideo.io
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor

from model import Generator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test Super Resolution')
    parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
    parser.add_argument('--video_name', type=str, help='low resolution video name')
    parser.add_argument('--model_name', default='netG_epoch_4_100.pth', type=str, help='super resolution model name')
    opt = parser.parse_args()

    UPSCALE_FACTOR = opt.upscale_factor
    VIDEO_NAME = opt.video_name
    MODEL_NAME = opt.model_name

    model = Generator(UPSCALE_FACTOR)
    if torch.cuda.is_available():
        model = model.cuda()
    # for cpu
    # model.load_state_dict(torch.load('epochs/' + MODEL_NAME, map_location=lambda storage, loc: storage))
    model.load_state_dict(torch.load('epochs/' + MODEL_NAME))

    video = skvideo.io.vreader(VIDEO_NAME)
    images = []
    for frame in video:
        image = Image.fromarray(frame)
        image = Variable(ToTensor()(image)).unsqueeze(dim=0)
        if torch.cuda.is_available():
            image = image.cuda()

        out = model(image).cpu().data[0].numpy() * 255
        out_img = out.astype(np.uint8)
        images.append(out_img)
    # save video
    skvideo.io.vwrite('out_' + VIDEO_NAME, np.array(images))
