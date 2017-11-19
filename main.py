import argparse

import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils import DatasetFromFolder
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
train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=64, shuffle=True)
val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=64, shuffle=False)

netG = Generator(UPSCALE_FACTOR)
print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
netD = Discriminator()
print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))
criterion = nn.BCELoss()
if torch.cuda.is_available():
    netD.cuda()
    netG.cuda()
    criterion.cuda()

optimizerD = optim.Adam(netD.parameters())
optimizerG = optim.Adam(netG.parameters())

for epoch in range(NUM_EPOCHS):
    bar = tqdm(train_loader)
    index = 1
    for data, target in bar:
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
        d_loss_real = criterion(real_out, real_label)
        real_scores = real_out.data.mean()

        # compute loss of fake_img
        z = Variable(data)
        if torch.cuda.is_available():
            z = z.cuda()
            fake_label = fake_label.cuda()
        fake_img = netG(z)
        fake_out = netD(fake_img)
        d_loss_fake = criterion(fake_out, fake_label)
        fake_scores = fake_out.data.mean()

        # bp and optimize
        d_loss = d_loss_real + d_loss_fake
        netD.zero_grad()
        d_loss.backward()
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        # compute loss of fake_img
        fake_img = netG(z)
        output = netD(fake_img)
        g_loss = criterion(output, real_label)

        # bp and optimize
        optimizerG.zero_grad()
        g_loss.backward()
        optimizerG.step()

        bar.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f'
                                 % (epoch + 1, NUM_EPOCHS, d_loss.data[0], g_loss.data[0], real_scores, fake_scores))

        if index % 100 == 0:
            utils.save_image(target, '%s/real_samples.png' % 'images', normalize=True)
            fake = netG(z)
            utils.save_image(fake.data, '%s/fake_samples_epoch_%d.png' % ('images', epoch), normalize=True)

    # save model parameters
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % ('epochs', epoch))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % ('epochs', epoch))
