import argparse
import os
from math import log10, fabs

import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils import DatasetFromFolder
from loss import GeneratorLoss, vgg16_relu2_2
from model import Discriminator, Generator

parser = argparse.ArgumentParser(description='Train Super Resolution')
parser.add_argument('--upscale_factor', default=3, type=int, help='super resolution upscale factor')
parser.add_argument('--num_epochs', default=100, type=int, help='super resolution epochs number')
opt = parser.parse_args()

UPSCALE_FACTOR = opt.upscale_factor
NUM_EPOCHS = opt.num_epochs
G_THRESHOLD = 0.1

train_set = DatasetFromFolder('data/train', upscale_factor=UPSCALE_FACTOR, input_transform=transforms.ToTensor(),
                              target_transform=transforms.ToTensor())
val_set = DatasetFromFolder('data/val', upscale_factor=UPSCALE_FACTOR, input_transform=transforms.ToTensor(),
                            target_transform=transforms.ToTensor())
train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=32, shuffle=True)
val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=32, shuffle=False)

netG = Generator(UPSCALE_FACTOR)
print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
netD = Discriminator()
print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))
# generator_criterion = GeneratorLoss(loss_network=vgg16_relu2_2())
generator_criterion = nn.BCELoss()
discriminator_criterion = nn.BCELoss()
if torch.cuda.is_available():
    netD.cuda()
    netG.cuda()
    generator_criterion.cuda()
    discriminator_criterion.cuda()

optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

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
        real_label = Variable(torch.ones(batch_size))
        fake_label = Variable(torch.zeros(batch_size))

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        real_img = Variable(target)
        if torch.cuda.is_available():
            real_img = real_img.cuda()
            real_label = real_label.cuda()

        # compute loss of real_img
        real_out = netD(real_img)
        d_loss_real = discriminator_criterion(real_out, real_label)
        real_scores = real_out.data.sum()
        running_real_scores += real_scores

        # compute loss of fake_img
        z = Variable(data)
        if torch.cuda.is_available():
            z = z.cuda()
            fake_label = fake_label.cuda()
        fake_img = netG(z)
        fake_out = netD(fake_img)
        d_loss_fake = discriminator_criterion(fake_out, fake_label)
        fake_scores = fake_out.data.sum()

        # bp and optimize
        d_loss = d_loss_real + d_loss_fake
        optimizerD.zero_grad()
        d_loss.backward(retain_graph=True)
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        while (fabs((real_scores - fake_scores) / batch_size) > G_THRESHOLD) or g_update_first:
            # compute loss of fake_img
            # g_loss = generator_criterion(fake_img, real_img, fake_out, real_label)
            g_loss = generator_criterion(fake_out, real_label)
            # bp and optimize
            optimizerG.zero_grad()
            g_loss.backward()
            optimizerG.step()
            fake_img = netG(z)
            fake_out = netD(fake_img)
            fake_scores = fake_out.data.sum()
            g_update_first = False

        g_loss = generator_criterion(fake_out, real_label)
        running_g_loss += g_loss.data[0] * batch_size
        d_loss_fake = discriminator_criterion(fake_out, fake_label)
        d_loss = d_loss_real + d_loss_fake
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
    valing_batch_sizes = 0
    for val_data, val_target in val_bar:
        batch_size = val_data.size(0)
        valing_batch_sizes += batch_size
        utils.save_image(val_target, out_path + 'HR_epoch_%d_batch_%d.png' % (epoch, index))
        lr = Variable(val_data)
        if torch.cuda.is_available():
            lr = lr.cuda()
        sr = netG(lr).data.cpu()
        utils.save_image(sr, out_path + 'SR_epoch_%d_batch_%d.png' % (epoch, index))
        batch_mse = ((sr - val_target) ** 2).mean()
        valing_mse += batch_mse * batch_size
        valing_psnr = 10 * log10(1 / (valing_mse / valing_batch_sizes))
        val_bar.set_description(desc='[convert LR images to SR images] PSNR: %.4f db' % valing_psnr)
        index += 1

    # save model parameters
    torch.save(netG.state_dict(), 'epochs/netG_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))
    torch.save(netD.state_dict(), 'epochs/netD_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))
