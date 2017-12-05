import argparse
import os
from math import log10

import pandas as pd
import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

import pytorch_ssim
from data_utils import DatasetFromFolder, train_hr_transform, val_hr_transform, lr_transform, upscale_transform
from loss import GeneratorLoss
from model import Generator, Discriminator

parser = argparse.ArgumentParser(description='Train Super Resolution')
parser.add_argument('--crop_size', default=72, type=int, help='super resolution crop size')
parser.add_argument('--upscale_factor', default=4, type=int, choices=[2, 3, 4, 8],
                    help='super resolution upscale factor')
parser.add_argument('--g_threshold', default=0.2, type=float, choices=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                    help='super resolution generator update threshold')
parser.add_argument('--g_stop_threshold', default=10, type=int, choices=[1, 10, 20, 30],
                    help='super resolution generator update stop threshold')
parser.add_argument('--num_epochs', default=100, type=int, help='super resolution epochs number')

opt = parser.parse_args()

CROP_SIZE = opt.crop_size
UPSCALE_FACTOR = opt.upscale_factor
NUM_EPOCHS = opt.num_epochs
G_THRESHOLD = opt.g_threshold
G_STOP_THRESHOLD = opt.g_stop_threshold

train_set = DatasetFromFolder('data/VOC2012/train', hr_transform=train_hr_transform(CROP_SIZE),
                              lr_transform=lr_transform(CROP_SIZE, UPSCALE_FACTOR))
val_set = DatasetFromFolder('data/VOC2012/val', hr_transform=val_hr_transform(CROP_SIZE),
                            lr_transform=lr_transform(CROP_SIZE, UPSCALE_FACTOR))
train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=64, shuffle=True)
val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=64, shuffle=False)

netG = Generator(UPSCALE_FACTOR)
print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
netD = Discriminator()
print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))

generator_criterion = GeneratorLoss()

if torch.cuda.is_available():
    netG.cuda()
    netD.cuda()
    generator_criterion.cuda()

optimizerG = optim.RMSprop(netG.parameters(), lr=1e-4)
optimizerD = optim.RMSprop(netD.parameters(), lr=1e-4)

results = {'d_loss': [], 'g_loss': [], 'psnr': [], 'ssim': []}

for epoch in range(1, NUM_EPOCHS + 1):
    train_bar = tqdm(train_loader)
    running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0}

    netG.train()
    netD.train()
    for data, target in train_bar:
        g_update_first = True
        batch_size = data.size(0)
        running_results['batch_sizes'] += batch_size

        ############################
        # (1) Update D network: minimize D(G(z)) - D(x)
        ###########################
        real_img = Variable(target)
        if torch.cuda.is_available():
            real_img = real_img.cuda()
        z = Variable(data)
        if torch.cuda.is_available():
            z = z.cuda()
        fake_img = netG(z)

        netD.zero_grad()
        real_out = netD(real_img).mean()
        fake_out = netD(fake_img).mean()
        d_loss = fake_out - real_out
        d_loss.backward(retain_graph=True)
        optimizerD.step()

        for p in netD.parameters():
            p.data.clamp_(-0.01, 0.01)

        ############################
        # (2) Update G network: minimize - D(G(z)) + Perception Loss + Image Loss + TV Loss
        ###########################
        index = 1
        while ((real_out.data[0] - fake_out.data[0] > G_THRESHOLD) or g_update_first) and (
                index <= G_STOP_THRESHOLD):
            netG.zero_grad()
            g_loss = generator_criterion(fake_out, fake_img, real_img)
            g_loss.backward()
            optimizerG.step()
            fake_img = netG(z)
            fake_out = netD(fake_img).mean()
            g_update_first = False
            index += 1

        g_loss = generator_criterion(fake_out, fake_img, real_img)
        running_results['g_loss'] += g_loss.data[0] * batch_size
        d_loss = fake_out - real_out
        running_results['d_loss'] += d_loss.data[0] * batch_size

        train_bar.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f' % (
            epoch, NUM_EPOCHS, running_results['d_loss'] / running_results['batch_sizes'],
            running_results['g_loss'] / running_results['batch_sizes']))

    netG.eval()
    out_path = 'images/SRF_' + str(UPSCALE_FACTOR) + '/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    val_bar = tqdm(val_loader)
    index = 1
    valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}

    for val_data, val_target in val_bar:
        batch_size = val_data.size(0)
        valing_results['batch_sizes'] += batch_size
        lr = Variable(val_data, volatile=True)
        hr = Variable(val_target, volatile=True)
        if torch.cuda.is_available():
            lr = lr.cuda()
            hr = hr.cuda()
        sr = netG(lr)

        image = utils.make_grid(torch.stack([upscale_transform(CROP_SIZE, UPSCALE_FACTOR)(lr), hr, sr], 1), nrow=3,
                                padding=5)
        utils.save_image(image.data.cpu(), out_path + 'epoch_%d_batch_%d.png' % (epoch, index), nrow=8, padding=5)

        batch_mse = ((sr - hr) ** 2).mean().data.cpu().numpy()
        valing_results['mse'] += batch_mse * batch_size
        batch_ssim = pytorch_ssim.ssim(sr, hr).data.cpu().numpy()
        valing_results['ssims'] += batch_ssim * batch_size
        valing_results['psnr'] = 10 * log10(1 / (valing_results['mse'] / valing_results['batch_sizes']))
        valing_results['ssim'] = valing_results['ssims'] / valing_results['batch_sizes']
        val_bar.set_description(
            desc='[convert LR images to SR images] PSNR: %.4f dB SSIM: %.4f' % (
                valing_results['psnr'], valing_results['ssim']))
        index += 1

    # save model parameters
    torch.save(netG.state_dict(), 'epochs/netG_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))
    torch.save(netD.state_dict(), 'epochs/netD_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))
    # save loss\scores\psnr\ssim
    results['g_loss'].append(running_results['g_loss'] / running_results['g_loss'])
    results['d_loss'].append(running_results['d_loss'] / running_results['batch_sizes'])
    results['psnr'].append(valing_results['psnr'])
    results['ssim'].append(valing_results['ssim'])

    if epoch % 10 == 0 and epoch != 0:
        out_path = 'statistics/SRF_' + str(UPSCALE_FACTOR) + '/'
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        data_frame = pd.DataFrame(
            data={'Loss_D': results['d_loss'], 'Loss_G': results['g_loss'], 'PSNR': results['psnr'],
                  'SSIM': results['ssim']}, index=range(1, epoch + 1))
        data_frame.to_csv(out_path + 'train_results.csv', index_label='Epoch')
