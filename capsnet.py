import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from functions import Routing


class CapsNet(nn.Module):
    def __init__(self, with_reconstruction=True):
        super(CapsNet, self).__init__()
        self.with_reconstruction = with_reconstruction

        self.conv1 = nn.Conv2d(1, 256, 9)
        self.primary_caps = nn.Conv2d(256, 32 * 8, 9, stride=2)
        self.digit_caps = Routing(32 * 6 * 6, 10, 8, 16, 32)

        if with_reconstruction:
            self.fc1 = nn.Linear(160, 512)
            self.fc2 = nn.Linear(512, 1024)
            self.fc3 = nn.Linear(1024, 784)

    def forward(self, x, target):
        # print(x.size())
        x = F.relu(self.conv1(x))
        # print(x.size())
        primary_caps = self.primary_caps(x)
        # print(primary_caps.size())
        digit_caps = self.digit_caps(primary_caps)
        # print(digit_caps.size())
        if self.with_reconstruction:
            mask = Variable(torch.zeros(digit_caps.size()))
            mask[:, target.data[0]] = digit_caps[:, target.data[0]]
            # print(mask)
            fc1 = F.relu(self.fc1(mask.view(-1)))
            fc2 = F.relu(self.fc2(fc1))
            reconstruction = F.sigmoid(self.fc3(fc2))
            return digit_caps, reconstruction
        return digit_caps


if __name__ == '__main__':
    net = CapsNet()
    print(net)
    d = torch.rand(3, 1, 28, 28)
    net(Variable(d), Variable(torch.LongTensor([3, 1, 2])))
