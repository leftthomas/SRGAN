import argparse
import os
from math import log10

import pandas as pd
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

import pytorch_ssim
from data_utils import DatasetFromFolder
from loss import GeneratorLoss
from model import Generator, Discriminator

parser = argparse.ArgumentParser(description='Train Super Resolution')
parser.add_argument('--upscale_factor', default=4, type=int, choices=[2, 3, 4, 8],
                    help='super resolution upscale factor')
parser.add_argument('--g_threshold', default=0.2, type=float, choices=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                    help='super resolution generator update threshold')
parser.add_argument('--g_stop_threshold', default=10, type=int, choices=[1, 10, 20, 30],
                    help='super resolution generator update stop threshold')
parser.add_argument('--num_epochs', default=100, type=int, help='super resolution epochs number')

opt = parser.parse_args()

UPSCALE_FACTOR = opt.upscale_factor
NUM_EPOCHS = opt.num_epochs
G_THRESHOLD = opt.g_threshold
G_STOP_THRESHOLD = opt.g_stop_threshold

train_set = DatasetFromFolder('data/train', upscale_factor=UPSCALE_FACTOR, input_transform=transforms.ToTensor(),
                              target_transform=transforms.ToTensor())
val_set = DatasetFromFolder('data/val', upscale_factor=UPSCALE_FACTOR, input_transform=transforms.ToTensor(),
                            target_transform=transforms.ToTensor())
train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=16, shuffle=True)
val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=16, shuffle=False)

netG = Generator(UPSCALE_FACTOR)
print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
netD = Discriminator()
print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))

generator_criterion = GeneratorLoss()

if torch.cuda.is_available():
    netG.cuda()
    netD.cuda()
    generator_criterion.cuda()

optimizerD = optim.Adam(netD.parameters())
optimizerG = optim.Adam(netG.parameters())
# optimizerD = optim.RMSprop(netD.parameters(), lr=1e-4)
# optimizerG = optim.RMSprop(netG.parameters(), lr=1e-4)


results = {'real_scores': [], 'fake_scores': [], 'd_loss': [], 'g_loss': [], 'psnr': [], 'ssim': []}

for epoch in range(1, NUM_EPOCHS + 1):
    train_bar = tqdm(train_loader)
    running_results = {'real_scores': 0, 'fake_scores': 0, 'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0}
    for data, target in train_bar:
        g_update_first = True
        batch_size = data.size(0)
        running_results['batch_sizes'] += batch_size

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        real_img = Variable(target)
        if torch.cuda.is_available():
            real_img = real_img.cuda()

        # compute score of real_img
        real_out = netD(real_img)
        real_scores = real_out.data.sum()
        running_results['real_scores'] += real_scores

        # compute score of fake_img
        z = Variable(data)
        if torch.cuda.is_available():
            z = z.cuda()
        fake_img = netG(z)
        fake_out = netD(fake_img)
        fake_scores = fake_out.data.sum()

        # compute loss of D network
        d_loss = - torch.mean(torch.log(real_out) + torch.log(1 - fake_out))

        # bp and optimize
        optimizerD.zero_grad()
        d_loss.backward(retain_graph=True)
        optimizerD.step()

        ############################
        # (2) Update G network: minimize log(1 - D(G(z))) + Perception Loss + Image Loss + TV Loss
        ###########################
        index = 1
        while (((real_scores - fake_scores) / batch_size > G_THRESHOLD) or g_update_first) and (
                    index <= G_STOP_THRESHOLD):
            # compute loss of G network
            g_loss = generator_criterion(fake_out, fake_img, real_img)
            # bp and optimize
            optimizerG.zero_grad()
            g_loss.backward()
            optimizerG.step()
            fake_img = netG(z)
            fake_out = netD(fake_img)
            fake_scores = fake_out.data.sum()
            g_update_first = False
            index += 1

        g_loss = generator_criterion(fake_out, fake_img, real_img)
        running_results['g_loss'] += g_loss.data[0] * batch_size
        d_loss = - torch.mean(torch.log(real_out) + torch.log(1 - fake_out))
        running_results['d_loss'] += d_loss.data[0] * batch_size
        running_results['fake_scores'] += fake_scores

        train_bar.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f'
                                       % (
                                           epoch, NUM_EPOCHS,
                                           running_results['d_loss'] / running_results['batch_sizes'],
                                           running_results['g_loss'] / running_results['batch_sizes'],
                                           running_results['real_scores'] / running_results['batch_sizes'],
                                           running_results['fake_scores'] / running_results['batch_sizes']))

    out_path = 'images/SRF_' + str(UPSCALE_FACTOR) + '/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    val_bar = tqdm(val_loader)
    index = 1
    valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}

    for val_data, val_target in val_bar:
        batch_size = val_data.size(0)
        valing_results['batch_sizes'] += batch_size
        if epoch == 1:
            utils.save_image(val_target, out_path + 'HR_batch_%d.png' % index, nrow=4, padding=5)
        lr = Variable(val_data, volatile=True)
        hr = Variable(val_target, volatile=True)
        if torch.cuda.is_available():
            lr = lr.cuda()
            hr = hr.cuda()
        sr = netG(lr)
        utils.save_image(sr.data.cpu(), out_path + 'SR_epoch_%d_batch_%d.png' % (epoch, index), nrow=4, padding=5)

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
    results['real_scores'].append(running_results['real_scores'] / running_results['batch_sizes'])
    results['fake_scores'].append(running_results['fake_scores'] / running_results['batch_sizes'])
    results['g_loss'].append(running_results['g_loss'] / running_results['g_loss'])
    results['d_loss'].append(running_results['d_loss'] / running_results['batch_sizes'])
    results['psnr'].append(valing_results['psnr'])
    results['ssim'].append(valing_results['ssim'])

    if epoch % 10 == 0 and epoch != 0:
        out_path = 'statistics/SRF_' + str(UPSCALE_FACTOR) + '/'
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        data_frame = pd.DataFrame(data=
                                  {'Loss_D': results['d_loss'], 'Loss_G': results['g_loss'],
                                   'D(x)': results['real_scores'],
                                   'D(G(z))': results['fake_scores'], 'PSNR': results['psnr'], 'SSIM': results['ssim']},
                                  index=range(1, epoch + 1))
        data_frame.to_csv(out_path + 'train_results.csv', index_label='Epoch')
