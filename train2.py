import math
import os

import torch
import torchvision.utils as vutils
from misc import PerceptualLoss, AvgMeter, LRTransformTest, TotalVariationLoss
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm

from model import Generator, Discriminator

train_args = {
    'hr_size': 72,  # make sure that hr_size can be divided by scale_factor exactly
    'scale_factor': 4,  # should be power of 2
    'g_lr': 1e-4,
    'd_lr': 1e-4,
    'train_set_path': 'data/train',
    'bsd100_path': 'data/BSD100',
    'set5_path': 'data/Set5',
    'set14_path': 'data/Set14',
    'ckpt_path': 'ckpt',
    'c': 0.01
}

train_ori_transform = transforms.Compose([
    transforms.RandomCrop(train_args['hr_size']),
    transforms.ToTensor()
])
train_lr_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Scale(train_args['hr_size'] // train_args['scale_factor'], interpolation=3),
    transforms.ToTensor()
])
val_ori_transform = transforms.ToTensor()
val_lr_transform = LRTransformTest(train_args['scale_factor'])
val_display_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Scale(400),
    transforms.CenterCrop(400),
    transforms.ToTensor()
])

train_set = datasets.ImageFolder(train_args['train_set_path'], train_ori_transform)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=12,
                          pin_memory=True)

bsd100 = datasets.ImageFolder(train_args['bsd100_path'], val_ori_transform)
set5 = datasets.ImageFolder(train_args['set5_path'], val_ori_transform)
set14 = datasets.ImageFolder(train_args['set14_path'], val_ori_transform)
bsd100_loader = DataLoader(bsd100, batch_size=1, num_workers=12, pin_memory=True)
set5_loader = DataLoader(set5, batch_size=1, num_workers=12, pin_memory=True)
set14_loader = DataLoader(set14, batch_size=1, num_workers=12, pin_memory=True)
val_loader = {'bsd100': bsd100_loader, 'set5': set5_loader, 'set14': set14_loader}


def train():
    g = Generator(scale_factor=train_args['scale_factor']).cuda().train()
    g = nn.DataParallel(g, device_ids=[0])
    mse_criterion = nn.MSELoss().cuda()

    d = Discriminator().cuda().train()
    d = nn.DataParallel(d, device_ids=[0])
    g_optimizer = optim.RMSprop(g.parameters(), lr=train_args['g_lr'])
    d_optimizer = optim.RMSprop(d.parameters(), lr=train_args['d_lr'])

    perceptual_criterion, tv_criterion = PerceptualLoss().cuda(), TotalVariationLoss().cuda()

    g_mse_loss_record, g_perceptual_loss_record, g_tv_loss_record = AvgMeter(), AvgMeter(), AvgMeter()
    psnr_record, g_ad_loss_record, g_loss_record, d_loss_record = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()

    for epoch in range(1, 101):
        train_bar = tqdm(train_loader)
        for data in train_bar:
            hr_imgs, _ = data
            batch_size = hr_imgs.size(0)
            lr_imgs = Variable(torch.stack([train_lr_transform(img) for img in hr_imgs], 0)).cuda()
            hr_imgs = Variable(hr_imgs).cuda()
            gen_hr_imgs = g(lr_imgs)

            # update d
            d.zero_grad()
            d_ad_loss = d(gen_hr_imgs.detach()).mean() - d(hr_imgs).mean()
            d_ad_loss.backward()
            d_optimizer.step()

            d_loss_record.update(d_ad_loss.data[0], batch_size)

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

            g_mse_loss_record.update(g_mse_loss.data[0], batch_size)
            g_perceptual_loss_record.update(g_perceptual_loss.data[0], batch_size)
            g_tv_loss_record.update(g_tv_loss.data[0], batch_size)
            psnr_record.update(10 * math.log10(1 / g_mse_loss.data[0]), batch_size)
            g_ad_loss_record.update(g_ad_loss.data[0], batch_size)
            g_loss_record.update(g_loss.data[0], batch_size)

            train_bar.set_description(
                '[train]: [epoch %d], [d_ad_loss %.5f], [g_ad_loss %.5f], [psnr %.5f], [g_loss %.5f]' % (
                    epoch, d_loss_record.avg, g_ad_loss_record.avg, psnr_record.avg, g_loss_record.avg))

        d_loss_record.reset()
        g_mse_loss_record.reset()
        g_perceptual_loss_record.reset()
        g_tv_loss_record.reset()
        psnr_record.reset()
        g_ad_loss_record.reset()
        g_loss_record.reset()

        validate(g, epoch, d)


def validate(g, curr_epoch, d):
    g.eval()

    mse_criterion = nn.MSELoss()
    g_mse_loss_record, psnr_record = AvgMeter(), AvgMeter()

    for name, loader in val_loader.items():

        val_visual = []
        # note that the batch size is 1
        val_bar = tqdm(loader)
        for data in val_bar:
            hr_img, _ = data

            lr_img, hr_restore_img = val_lr_transform(hr_img.squeeze(0))

            lr_img = Variable(lr_img.unsqueeze(0), volatile=True).cuda()
            hr_restore_img = hr_restore_img
            hr_img = Variable(hr_img, volatile=True).cuda()

            gen_hr_img = g(lr_img)

            g_mse_loss = mse_criterion(gen_hr_img, hr_img)

            g_mse_loss_record.update(g_mse_loss.data[0])
            psnr_record.update(10 * math.log10(1 / g_mse_loss.data[0]))

            val_visual.extend([val_display_transform(hr_restore_img),
                               val_display_transform(hr_img.cpu().data.squeeze(0)),
                               val_display_transform(gen_hr_img.cpu().data.squeeze(0))])

        val_visual = torch.stack(val_visual, 0)
        val_visual = vutils.make_grid(val_visual, nrow=3, padding=5)
        vutils.save_image(val_visual, 'results/SR_epoch_%d_data_%s.png' % (curr_epoch, name), nrow=2)

        snapshot_name = 'epoch_%d_%s_g_mse_loss_%.5f_psnr_%.5f' % (
            curr_epoch, name, g_mse_loss_record.avg, psnr_record.avg)

        print('[validate %s]: [epoch %d], [g_mse_loss %.5f], [psnr %.5f]' % (
            name, curr_epoch, g_mse_loss_record.avg, psnr_record.avg))

        torch.save(d.state_dict(), os.path.join(train_args['ckpt_path'], snapshot_name + '_d.pth'))

        torch.save(g.state_dict(), os.path.join(train_args['ckpt_path'], snapshot_name + '_g.pth'))

        g_mse_loss_record.reset()
        psnr_record.reset()

    g.train()


if __name__ == '__main__':
    train()
