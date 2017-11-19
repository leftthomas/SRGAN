import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import argparse
from data_utils import DatasetFromFolder
import torchvision.transforms as transforms
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model import Discriminator, Generator
from tqdm import tqdm

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
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        netD.zero_grad()
        real_label = torch.Tensor(batch_size * [1])
        if torch.cuda.is_available():
            target = target.cuda()
            real_label = real_label.cuda()
        real = Variable(target)
        real_label = Variable(real_label)

        real_output = netD(real)
        errD_real = criterion(real_output, real_label)
        errD_real.backward()
        D_x = real_output.data.mean()

        # train with fake
        fake_label = torch.Tensor(batch_size * [0])
        if torch.cuda.is_available():
            data = data.cuda()
            fake_label = fake_label.cuda()
        fake = Variable(data)
        fake = netG(fake)
        fake_label = Variable(fake_label)
        fake_output = netD(fake.detach())
        errD_fake = criterion(fake_output, fake_label)
        errD_fake.backward()
        D_G_z1 = fake_output.data.mean()

        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        # fake labels are real for generator cost
        label = torch.Tensor(batch_size * [1])
        if torch.cuda.is_available():
            label = label.cuda()
        output = netD(fake)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.data.mean()
        optimizerG.step()

        bar.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                                 % (epoch + 1, NUM_EPOCHS, errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2))

        if index % 100 == 0:
            utils.save_image(target, '%s/real_samples.png' % 'images', normalize=True)
            fake = netG(Variable(data))
            utils.save_image(fake.data, '%s/fake_samples_epoch_%d.png' % ('images', epoch), normalize=True)

    # save model parameters
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % ('epochs', epoch))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % ('epochs', epoch))
