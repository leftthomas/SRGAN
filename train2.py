import torch
from misc import PerceptualLoss, TotalVariationLoss
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

train_set = datasets.ImageFolder(train_args['train_set_path'], train_ori_transform)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=12,
                          pin_memory=True)


def train():
    g = Generator(scale_factor=train_args['scale_factor']).cuda().train()
    g = nn.DataParallel(g, device_ids=[0])
    mse_criterion = nn.MSELoss().cuda()

    d = Discriminator().cuda().train()
    d = nn.DataParallel(d, device_ids=[0])
    g_optimizer = optim.RMSprop(g.parameters(), lr=train_args['g_lr'])
    d_optimizer = optim.RMSprop(d.parameters(), lr=train_args['d_lr'])

    perceptual_criterion, tv_criterion = PerceptualLoss().cuda(), TotalVariationLoss().cuda()

    for epoch in range(1, 101):
        train_bar = tqdm(train_loader)
        for data in train_bar:
            hr_imgs, _ = data
            lr_imgs = Variable(torch.stack([train_lr_transform(img) for img in hr_imgs], 0)).cuda()
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


if __name__ == '__main__':
    train()
