import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable


def squash(tensor, dim=1):
    squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1 + squared_norm)
    return scale * tensor / torch.sqrt(squared_norm)


class CapsuleGenerator(nn.Module):
    def __init__(self, upscale_factor):
        super(CapsuleGenerator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=4, dilation=2)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=3, dilation=3,
                               groups=16)
        self.conv4 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=3, dilation=3,
                               groups=8)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=5, stride=1, padding=4, dilation=2,
                               groups=4)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=3 * (upscale_factor ** 2), kernel_size=7, stride=1,
                               padding=3)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.lrelu(self.conv1(x))
        x = self.lrelu(self.conv2(x))

        x = self.conv3(x)
        # capsules squash
        x = torch.cat([squash(capsule) for capsule in torch.chunk(x, chunks=16, dim=1)], dim=1)
        x = self.conv4(x)
        x = torch.cat([squash(capsule) for capsule in torch.chunk(x, chunks=8, dim=1)], dim=1)
        x = self.conv5(x)
        x = torch.cat([squash(capsule) for capsule in torch.chunk(x, chunks=4, dim=1)], dim=1)
        x = self.conv6(x)

        x = F.sigmoid(self.pixel_shuffle(x))
        return x


class CapsuleDiscriminator(nn.Module):
    def __init__(self):
        super(CapsuleDiscriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=4, dilation=2)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=3, dilation=3,
                               groups=16)
        self.conv4 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=3, dilation=3,
                               groups=8)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=5, stride=1, padding=4, dilation=2,
                               groups=4)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=3, kernel_size=7, stride=1,
                               padding=3)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.lrelu(self.conv1(x))
        x = self.lrelu(self.conv2(x))

        x = self.conv3(x)
        # capsules squash
        x = torch.cat([squash(capsule) for capsule in torch.chunk(x, chunks=16, dim=1)], dim=1)
        x = self.conv4(x)
        x = torch.cat([squash(capsule) for capsule in torch.chunk(x, chunks=8, dim=1)], dim=1)
        x = self.conv5(x)
        x = torch.cat([squash(capsule) for capsule in torch.chunk(x, chunks=4, dim=1)], dim=1)
        x = self.conv6(x)

        x = x.view(x.size(0), -1).norm(dim=-1)
        return F.sigmoid(x)


if __name__ == "__main__":
    a = torch.FloatTensor([[0, 1, 2], [3, 4, 5]])
    b = squash(a)
    print(b)
    images = Variable(torch.randn(10, 3, 75, 75))
    g = CapsuleGenerator(upscale_factor=3)
    print(g)
    output = g(images)
    d = CapsuleDiscriminator()
    print(d)
    prob = d(output)
    print(prob)
