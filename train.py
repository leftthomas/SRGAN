import argparse
import os
from math import log10

import torch.nn as nn
import torch.nn.functional as F
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
generator_criterion = GeneratorLoss(loss_network=vgg16_relu2_2())
discriminator_criterion = nn.BCELoss()
if torch.cuda.is_available():
    netD.cuda()
    netG.cuda()
    generator_criterion.cuda()
    discriminator_criterion.cuda()

optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

for epoch in range(NUM_EPOCHS):
    train_bar = tqdm(train_loader)
    for data, target in train_bar:
        batch_size = data.size(0)
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
        real_scores = real_out.data.mean()

        # compute loss of fake_img
        z = Variable(data)
        if torch.cuda.is_available():
            z = z.cuda()
            fake_label = fake_label.cuda()
        fake_img = netG(z)
        fake_out = netD(fake_img)
        d_loss_fake = discriminator_criterion(fake_out, fake_label)
        fake_scores = fake_out.data.mean()

        # bp and optimize
        d_loss = d_loss_real + d_loss_fake
        optimizerD.zero_grad()
        d_loss.backward(retain_graph=True)
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        # compute loss of fake_img
        g_loss = generator_criterion(fake_img, real_img, fake_out, real_label)

        # bp and optimize
        optimizerG.zero_grad()
        g_loss.backward()
        optimizerG.step()

        train_bar.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f'
                                       % (
                                           epoch + 1, NUM_EPOCHS, d_loss.data[0], g_loss.data[0], real_scores,
                                           fake_scores))

    out_path = 'images/SRF_' + str(UPSCALE_FACTOR) + '/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    val_bar = tqdm(val_loader)
    index = 1
    for val_data, val_target in val_bar:
        utils.save_image(val_target, out_path + 'HR_epoch_%d_batch_%d.png' % (epoch + 1, index))
        lr = Variable(val_data)
        if torch.cuda.is_available():
            lr = lr.cuda()
        sr = netG(lr).data.cpu()
        utils.save_image(sr, out_path + 'SR_epoch_%d_batch_%d.png' % (epoch + 1, index))
        mse = F.mse_loss(sr, val_target)
        psnr = 10 * log10(1 / mse.data[0])
        val_bar.set_description(desc='[convert LR images to SR images] PSNR: %.4f db' % psnr)
        index += 1

    # save model parameters
    torch.save(netG.state_dict(), 'epochs/netG_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))
    torch.save(netD.state_dict(), 'epochs/netD_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))
