import argparse

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage

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
    size1 = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH) * UPSCALE_FACTOR),
             int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)) * UPSCALE_FACTOR)
    size2 = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH) * UPSCALE_FACTOR * 2 + 10),
             int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)) * UPSCALE_FACTOR + 10 + int(
                 int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH) * UPSCALE_FACTOR * 2 + 10) / int(
                     10 * int(int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH) * UPSCALE_FACTOR) // 5 + 1)) * int(
                     int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH) * UPSCALE_FACTOR) // 5 - 9)))
    videoWriter = cv2.VideoWriter('out_' + VIDEO_NAME, cv2.VideoWriter_fourcc(*'MPEG'), fps, size1)
    videoWriter2 = cv2.VideoWriter('compare_' + VIDEO_NAME, cv2.VideoWriter_fourcc(*'MPEG'), fps, size2)
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
        # save sr video
        videoWriter.write(out_img)

        out_img = ToPILImage()(out_img)
        crop_out_imgs = transforms.FiveCrop(size=out_img.width // 5 - 9)(out_img)
        crop_out_imgs = [np.asarray(transforms.Pad(padding=(10, 5, 0, 0))(img)) for img in crop_out_imgs]
        out_img = transforms.Pad(padding=(5, 0, 0, 5))(out_img)
        compared_img = transforms.Resize(size=(size1[1], size1[0]), interpolation=Image.BICUBIC)(ToPILImage()(frame))
        crop_compared_imgs = transforms.FiveCrop(size=compared_img.width // 5 - 9)(compared_img)
        crop_compared_imgs = [np.asarray(transforms.Pad(padding=(0, 5, 10, 0))(img)) for img in crop_compared_imgs]
        compared_img = transforms.Pad(padding=(0, 0, 5, 5))(compared_img)

        top_image = np.concatenate((np.asarray(compared_img), np.asarray(out_img)), axis=1)
        bottom_image = np.concatenate(crop_compared_imgs + crop_out_imgs, axis=1)
        bottom_image = np.asarray(transforms.Resize(
            size=(int(top_image.shape[1] / bottom_image.shape[1] * bottom_image.shape[0]), top_image.shape[1]))(
            ToPILImage()(bottom_image)))
        final_image = np.concatenate((top_image, bottom_image))
        # save compared video
        videoWriter2.write(final_image)
        # next frame
        success, frame = videoCapture.read()
