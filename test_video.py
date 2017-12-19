import argparse
import os
from os import listdir

import cv2
import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage
from tqdm import tqdm

from data_utils import is_video_file
from model import Generator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test Super Resolution')
    parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
    parser.add_argument('--model_name', default='netG_epoch_4_100.pth', type=str, help='super resolution model name')
    opt = parser.parse_args()

    UPSCALE_FACTOR = opt.upscale_factor
    MODEL_NAME = opt.model_name

    path = 'data/test/SRF_' + str(UPSCALE_FACTOR) + '/video/'
    videos_name = [x for x in listdir(path) if is_video_file(x)]
    model = Generator(UPSCALE_FACTOR)
    if torch.cuda.is_available():
        model = model.cuda()
    # for cpu
    # model.load_state_dict(torch.load('epochs/' + MODEL_NAME, map_location=lambda storage, loc: storage))
    model.load_state_dict(torch.load('epochs/' + MODEL_NAME))

    out_path = 'results/SRF_' + str(UPSCALE_FACTOR) + '/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    for video_name in tqdm(videos_name, desc='convert LR videos to HR videos'):
        videoCapture = cv2.VideoCapture(path + video_name)
        fps = videoCapture.get(cv2.CAP_PROP_FPS)
        size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH) * UPSCALE_FACTOR),
                int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)) * UPSCALE_FACTOR)
        output_name = out_path + video_name.split('.')[0] + '.avi'
        videoWriter = cv2.VideoWriter(output_name, cv2.VideoWriter_fourcc(*'MPEG'), fps, size)
        # read frame
        success, frame = videoCapture.read()
        while success:
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image = Variable(ToTensor()(image)).unsqueeze(dim=0)
            if torch.cuda.is_available():
                image = image.cuda()

            out = model(image).cpu().data[0]
            out_img = ToPILImage()(out)
            out_img.show()
            out_img = cv2.cvtColor(np.asarray(out_img), cv2.COLOR_RGB2BGR)
            # save video
            videoWriter.write(out_img)
            # next frame
            success, frame = videoCapture.read()
