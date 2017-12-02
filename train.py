import argparse
import os
from math import log10, fabs

import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

import pytorch_ssim
from data_utils import DatasetFromFolder
from loss import GeneratorAdversarialLoss, vgg19_loss_network, GeneratorAdversarialWithContentLoss, \
    GeneratorAdversarialWithPixelMSELoss, PerceptualLoss, TotalVariationLoss, vgg16_loss_network
from model import Generator, Discriminator

parser = argparse.ArgumentParser(description='Train Super Resolution')
parser.add_argument('--upscale_factor', default=4, type=int, choices=[2, 3, 4, 8],
                    help='super resolution upscale factor')
parser.add_argument('--g_threshold', default=0.2, type=float, choices=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                    help='super resolution generator update threshold')
parser.add_argument('--g_stop_threshold', default=10, type=int, choices=[1, 10, 20, 30],
                    help='super resolution generator update stop threshold')
parser.add_argument('--g_loss_type', default='GACL', type=str,
                    choices=['GAL', 'GAML', 'GACL', 'GACLL', 'GACLV', 'GACLLV'],
                    help='super resolution generator loss function type')
parser.add_argument('--num_epochs', default=100, type=int, help='super resolution epochs number')

opt = parser.parse_args()

UPSCALE_FACTOR = opt.upscale_factor
NUM_EPOCHS = opt.num_epochs
G_THRESHOLD = opt.g_threshold
G_STOP_THRESHOLD = opt.g_stop_threshold
G_LOSS_TYPE = opt.g_loss_type

train_set = DatasetFromFolder('data/train', upscale_factor=UPSCALE_FACTOR, input_transform=transforms.ToTensor(),
                              target_transform=transforms.ToTensor())
val_set = DatasetFromFolder('data/val', upscale_factor=UPSCALE_FACTOR, input_transform=transforms.ToTensor(),
                            target_transform=transforms.ToTensor())
train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=64, shuffle=True, pin_memory=True)
val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=64, shuffle=False, pin_memory=True)

netG = Generator(UPSCALE_FACTOR)
print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
netD = Discriminator()
print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))

if G_LOSS_TYPE == 'GAL':
    generator_criterion = GeneratorAdversarialLoss()
elif G_LOSS_TYPE == 'GAML':
    generator_criterion = GeneratorAdversarialWithPixelMSELoss()
elif G_LOSS_TYPE == 'GACL':
    generator_criterion = GeneratorAdversarialWithContentLoss(loss_network=vgg19_loss_network(is_last=True),
                                                              using_l1=False)
elif G_LOSS_TYPE == 'GACLL':
    generator_criterion = GeneratorAdversarialWithContentLoss(loss_network=vgg19_loss_network(is_last=True),
                                                              using_l1=True)
elif G_LOSS_TYPE == 'GACLV':
    generator_criterion = GeneratorAdversarialWithContentLoss(loss_network=vgg19_loss_network(is_last=False),
                                                              using_l1=False)
else:
    # G_LOSS_TYPE == 'GACLLV'
    generator_criterion = GeneratorAdversarialWithContentLoss(loss_network=vgg19_loss_network(is_last=False),
                                                              using_l1=True)

perceptual_criterion, tv_criterion, mse_criterion = PerceptualLoss(
    vgg16_loss_network()), TotalVariationLoss(), nn.MSELoss()

if torch.cuda.is_available():
    netG = netG.cuda()
    netD = netD.cuda()
    generator_criterion = generator_criterion.cuda()
    perceptual_criterion = perceptual_criterion.cuda()
    tv_criterion = tv_criterion.cuda()
    mse_criterion = mse_criterion.cuda()

optimizerD = optim.RMSprop(netD.parameters(), lr=1e-4)
optimizerG = optim.RMSprop(netG.parameters(), lr=1e-4)

results_real_scores = []
results_fake_scores = []
results_d_loss = []
results_g_loss = []
results_psnr = []
results_ssim = []
for epoch in range(1, NUM_EPOCHS + 1):
    train_bar = tqdm(train_loader)
    running_real_scores = 0
    running_fake_scores = 0
    running_batch_sizes = 0
    running_d_loss = 0
    running_g_loss = 0
    for data, target in train_bar:
        g_update_first = True
        batch_size = data.size(0)
        running_batch_sizes += batch_size

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        real_img = Variable(target)
        if torch.cuda.is_available():
            real_img = real_img.cuda()

        # compute loss of real_img
        real_out = netD(real_img)
        real_scores = real_out.data.sum()
        running_real_scores += real_scores

        # compute loss of fake_img
        z = Variable(data)
        if torch.cuda.is_available():
            z = z.cuda()
        fake_img = netG(z)
        fake_out = netD(fake_img)
        fake_scores = fake_out.data.sum()

        # d_loss = - torch.mean(torch.log(real_out) + torch.log(1 - fake_out))
        d_loss = fake_out.mean() - real_out.mean()

        # bp and optimize
        optimizerD.zero_grad()
        d_loss.backward(retain_graph=True)
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        index = 1
        while ((fabs((real_scores - fake_scores) / batch_size) > G_THRESHOLD) or g_update_first) and (
                    index <= G_STOP_THRESHOLD):
            # compute loss of fake_img
            g_mse_loss = mse_criterion(fake_img, real_img)
            g_perceptual_loss = perceptual_criterion(fake_img, real_img)
            g_tv_loss = tv_criterion(fake_img)
            g_ad_loss = -netD(fake_img).mean()
            g_loss = g_mse_loss + 0.006 * g_perceptual_loss + 2e-8 * g_tv_loss + 0.001 * g_ad_loss

            # g_loss = generator_criterion(fake_out, fake_img, real_img)
            # bp and optimize
            optimizerG.zero_grad()
            g_loss.backward()
            optimizerG.step()
            fake_img = netG(z)
            fake_out = netD(fake_img)
            fake_scores = fake_out.data.sum()
            g_update_first = False
            index += 1

        g_mse_loss = mse_criterion(fake_img, real_img)
        g_perceptual_loss = perceptual_criterion(fake_img, real_img)
        g_tv_loss = tv_criterion(fake_img)
        g_ad_loss = -netD(fake_img).mean()
        g_loss = g_mse_loss + 0.006 * g_perceptual_loss + 2e-8 * g_tv_loss + 0.001 * g_ad_loss

        # g_loss = generator_criterion(fake_out, fake_img, real_img)
        running_g_loss += g_loss.data[0] * batch_size
        d_loss = - torch.mean(torch.log(real_out) + torch.log(1 - fake_out))
        running_d_loss += d_loss.data[0] * batch_size
        running_fake_scores += fake_scores

        train_bar.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f'
                                       % (
                                           epoch, NUM_EPOCHS, running_d_loss / running_batch_sizes,
                                           running_g_loss / running_batch_sizes,
                                           running_real_scores / running_batch_sizes,
                                           running_fake_scores / running_batch_sizes))

    out_path = 'images/SRF_' + str(UPSCALE_FACTOR) + '/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    val_bar = tqdm(val_loader)
    index = 1
    valing_mse = 0
    valing_ssims = 0
    valing_psnr = 0
    valing_ssim = 0
    valing_batch_sizes = 0
    for val_data, val_target in val_bar:
        batch_size = val_data.size(0)
        valing_batch_sizes += batch_size
        if epoch == 1:
            utils.save_image(val_target, out_path + 'HR_batch_%d.png' % index, nrow=8, padding=5)
        lr = Variable(val_data, volatile=True)
        hr = Variable(val_target, volatile=True)
        if torch.cuda.is_available():
            lr = lr.cuda()
            hr = hr.cuda()
        sr = netG(lr)
        utils.save_image(sr.data.cpu(), out_path + 'SR_epoch_%d_batch_%d.png' % (epoch, index), nrow=8, padding=5)

        batch_mse = ((sr - hr) ** 2).mean().data.cpu().numpy()
        valing_mse += batch_mse * batch_size
        batch_ssim = pytorch_ssim.ssim(sr, hr).data.cpu().numpy()
        valing_ssims += batch_ssim * batch_size
        valing_psnr = 10 * log10(1 / (valing_mse / valing_batch_sizes))
        valing_ssim = valing_ssims / valing_batch_sizes
        val_bar.set_description(
            desc='[convert LR images to SR images] PSNR: %.4f dB SSIM: %.4f' % (valing_psnr, valing_ssim))
        index += 1

    # save model parameters
    torch.save(netG.state_dict(), 'epochs/netG_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))
    torch.save(netD.state_dict(), 'epochs/netD_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))
    # save loss\scores\psnr\ssim
    results_real_scores.append(running_real_scores / running_batch_sizes)
    results_fake_scores.append(running_fake_scores / running_batch_sizes)
    results_g_loss.append(running_g_loss / running_batch_sizes)
    results_d_loss.append(running_d_loss / running_batch_sizes)
    results_psnr.append(valing_psnr)
    results_ssim.append(valing_ssim)

    if epoch % 10 == 0 and epoch != 0:
        out_path = 'statistics/SRF_' + str(UPSCALE_FACTOR) + '/'
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        data_frame = pd.DataFrame(data=
                                  {'Loss_D': results_d_loss, 'Loss_G': results_g_loss,
                                   'D(x)': results_real_scores,
                                   'D(G(z))': results_fake_scores, 'PSNR': results_psnr, 'SSIM': results_ssim},
                                  index=range(1, epoch + 1))
        data_frame.to_csv(out_path + 'train_results.csv', index_label='Epoch')
