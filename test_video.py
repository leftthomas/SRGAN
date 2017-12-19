import argparse

import cv2
import numpy as np
import torch
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

    model = Generator(UPSCALE_FACTOR).eval()
    if torch.cuda.is_available():
        model = model.cuda()
    # for cpu
    # model.load_state_dict(torch.load('epochs/' + MODEL_NAME, map_location=lambda storage, loc: storage))
    model.load_state_dict(torch.load('epochs/' + MODEL_NAME))

    videoCapture = cv2.VideoCapture(VIDEO_NAME)
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH) * UPSCALE_FACTOR),
            int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)) * UPSCALE_FACTOR)
    videoWriter = cv2.VideoWriter('out_' + VIDEO_NAME, cv2.VideoWriter_fourcc(*'MPEG'), fps, size)
    # read frame
    success, frame = videoCapture.read()
    while success:
        image = Variable(ToTensor()(frame), volatile=True).unsqueeze(0)
        if torch.cuda.is_available():
            image = image.cuda()

        out = model(image)
        out = out.cpu()
        out_img = out.data[0].numpy()
        out_img *= 255.0
        out_img = (np.uint8(out_img)).transpose((1, 2, 0))
        # save video
        videoWriter.write(out_img)
        # next frame
        success, frame = videoCapture.read()
