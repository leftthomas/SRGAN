import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, upscale_factor):
        super(Generator, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(32, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv5 = nn.Conv2d(32, 3 * (upscale_factor ** 2), (3, 3), (1, 1), (1, 1))
        self.prelu = nn.PReLU()
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.prelu(self.conv1(x), inplace=True)
        x = self.prelu(self.conv2(x), inplace=True)
        x = self.prelu(self.conv3(x), inplace=True)
        x = self.prelu(self.conv4(x), inplace=True)
        x = F.sigmoid(self.pixel_shuffle(self.conv5(x)))
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 32 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32 * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32 * 2, 1, 2, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        output = self.main(x)
        return output.view(output.size(0), -1).mean(dim=-1)


if __name__ == "__main__":
    netG = Generator(upscale_factor=3)
    print(netG)
    netD = Discriminator()
    print(netD)
