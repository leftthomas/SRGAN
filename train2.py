import math
import os

import torch
import torchvision.utils as vutils
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm

from data_utils import DatasetFromFolder, LRTransformTest, AvgMeter
from loss import GeneratorLoss
from model import Generator, Discriminator

train_args = {
    'train_batch_size': 64,
    'scale_factor': 4,  # should be power of 2
    'ckpt_path': 'epochs',
}


val_lr_transform = LRTransformTest(train_args['scale_factor'])
val_display_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Scale(train_args['scale_factor'] * 100),
    transforms.CenterCrop(train_args['scale_factor'] * 100),
    transforms.ToTensor()
])

train_set = DatasetFromFolder('data/VOC2012/train', train_args['scale_factor'])
train_loader = DataLoader(train_set, batch_size=train_args['train_batch_size'], shuffle=True, num_workers=12,
                          pin_memory=True)
val_set = datasets.ImageFolder('../SRGAN/data/BSD100', transforms.ToTensor())
val_loader = DataLoader(val_set, batch_size=1, num_workers=12, pin_memory=True)


def train():
    g = Generator(scale_factor=train_args['scale_factor']).cuda().train()
    d = Discriminator().cuda().train()

    g_optimizer = optim.RMSprop(g.parameters(), lr=1e-4)
    d_optimizer = optim.RMSprop(d.parameters(), lr=1e-4)

    generator_criterion = GeneratorLoss().cuda()

    g_loss_record, d_loss_record = AvgMeter(), AvgMeter()

    for epoch in range(1, 101):
        train_bar = tqdm(train_loader)
        for lr_imgs, hr_imgs in train_bar:
            batch_size = hr_imgs.size(0)
            lr_imgs = Variable(lr_imgs).cuda()
            hr_imgs = Variable(hr_imgs).cuda()
            gen_hr_imgs = g(lr_imgs)

            # update d
            d.zero_grad()
            d_loss = d(gen_hr_imgs.detach()).mean() - d(hr_imgs).mean()
            d_loss.backward()
            d_optimizer.step()

            d_loss_record.update(d_loss.data[0], batch_size)

            for p in d.parameters():
                p.data.clamp_(-0.01, 0.01)

            # update g
            g.zero_grad()
            g_loss = generator_criterion(d(gen_hr_imgs), gen_hr_imgs, hr_imgs)
            g_loss.backward()
            g_optimizer.step()

            g_loss_record.update(g_loss.data[0], batch_size)

            train_bar.set_description(
                '[train]: [epoch %d], [d_loss %.5f], [g_loss %.5f]' % (epoch, d_loss_record.avg, g_loss_record.avg))

        d_loss_record.reset()
        g_loss_record.reset()

        validate(g, epoch, d)


def validate(g, curr_epoch, d):
    g.eval()

    mse_criterion = nn.MSELoss()
    g_mse_loss_record, psnr_record = AvgMeter(), AvgMeter()

    val_visual = []
    # note that the batch size is 1
    for data in val_loader:
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
    val_visual = torch.chunk(val_visual, len(val_visual) // 8)
    for i, visual in enumerate(val_visual):
        visual = vutils.make_grid(visual, nrow=3, padding=5)
        vutils.save_image(visual, 'images/' + str(curr_epoch) + '_' + str(i) + '.png')

    snapshot_name = 'epoch_%d_g_mse_loss_%.5f_psnr_%.5f' % (curr_epoch, g_mse_loss_record.avg, psnr_record.avg)

    print(
        '[validate]: [epoch %d], [g_mse_loss %.5f], [psnr %.5f]' % (curr_epoch, g_mse_loss_record.avg, psnr_record.avg))

    torch.save(d.state_dict(), os.path.join(train_args['ckpt_path'], snapshot_name + '_d.pth'))

    torch.save(g.state_dict(), os.path.join(train_args['ckpt_path'], snapshot_name + '_g.pth'))

    g_mse_loss_record.reset()
    psnr_record.reset()

    g.train()


if __name__ == '__main__':
    train()
