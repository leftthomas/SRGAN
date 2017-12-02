import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from data_utils import DatasetFromFolder
from loss import PerceptualLoss, TotalVariationLoss, vgg16_loss_network
from model import Generator, Discriminator

train_args = {
    'scale_factor': 4,  # should be power of 2
    'c': 0.01
}

train_set = DatasetFromFolder('data/train', upscale_factor=train_args['scale_factor'],
                              input_transform=transforms.ToTensor(),
                              target_transform=transforms.ToTensor())
train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=4)


def train():
    g = Generator(scale_factor=train_args['scale_factor']).cuda().train()
    mse_criterion = nn.MSELoss().cuda()

    d = Discriminator().cuda().train()
    g_optimizer = optim.RMSprop(g.parameters(), lr=1e-4)
    d_optimizer = optim.RMSprop(d.parameters(), lr=1e-4)

    perceptual_criterion, tv_criterion = PerceptualLoss(vgg16_loss_network()).cuda(), TotalVariationLoss().cuda()

    for epoch in range(1, 101):
        train_bar = tqdm(train_loader)
        for lr_imgs, hr_imgs in train_bar:
            lr_imgs = Variable(lr_imgs).cuda()
            hr_imgs = Variable(hr_imgs).cuda()
            gen_hr_imgs = g(lr_imgs)

            # update d
            d.zero_grad()
            d_ad_loss = d(gen_hr_imgs.detach()).mean() - d(hr_imgs).mean()
            d_ad_loss.backward()
            d_optimizer.step()

            for p in d.parameters():
                p.data.clamp_(-train_args['c'], train_args['c'])

            # update g
            g.zero_grad()
            g_mse_loss = mse_criterion(gen_hr_imgs, hr_imgs)
            g_perceptual_loss = perceptual_criterion(gen_hr_imgs, hr_imgs)
            g_tv_loss = tv_criterion(gen_hr_imgs)
            g_ad_loss = -d(gen_hr_imgs).mean()
            g_loss = g_mse_loss + 0.006 * g_perceptual_loss + 2e-8 * g_tv_loss + 0.001 * g_ad_loss
            g_loss.backward()
            g_optimizer.step()

        # save model parameters
        torch.save(g.state_dict(), 'epochs/netG_epoch_%d_%d.pth' % (train_args['scale_factor'], epoch))
        torch.save(d.state_dict(), 'epochs/netD_epoch_%d_%d.pth' % (train_args['scale_factor'], epoch))


if __name__ == '__main__':
    train()
