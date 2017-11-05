import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class DigitMarginLoss(nn.Module):
    def __init__(self, m_plus=0.9, m_minus=0.1, lamda=0.5):
        super(DigitMarginLoss, self).__init__()
        self.m_plus = m_plus
        self.m_minus = m_minus
        self.lamda = lamda

    def forward(self, output, target):
        norm = output.norm(dim=0)
        zero = Variable(torch.zeros(1))
        losses = [torch.max(zero, self.m_plus - norm).pow(2) if digit == target.data[0]
                  else self.lamda * torch.max(zero, norm - self.m_minus).pow(2)
                  for digit in range(10)]
        return torch.cat(losses).sum()


def squash(vec):
    norm = vec.norm(dim=1, keepdim=True)
    norm_squared = norm ** 2
    coeff = norm_squared / (1 + norm_squared)
    return (coeff / norm) * vec


def accuracy(output, target):
    pred = output.norm(dim=0).max(0)[1].data[0]
    target = target.data[0]
    return int(pred == target)


class Routing(nn.Module):
    def __init__(self, num_in_caps, num_out_caps, in_dim, out_dim, num_shared):
        super(Routing, self).__init__()
        self.in_dim = in_dim
        self.num_shared = num_shared

        self.W = [nn.Linear(in_dim, out_dim, bias=False) for _ in range(num_shared)]
        self.b = Variable(torch.zeros(num_in_caps, num_out_caps))

    def forward(self, x):
        batch_size, in_channels, h, w = x.size()

        for index in range(batch_size):
            y = x[index]
            y = y.view(self.num_shared, -1, self.in_dim)
            groups = y.chunk(self.num_shared)
            u = [group.squeeze().chunk(h * w) for group in groups]
            pred = [self.W[i](in_vec.squeeze()) for i, group in enumerate(u) for in_vec in group]
            pred = torch.stack([torch.stack(p) for p in pred]).view(self.num_shared * h * w, -1)
            c = F.softmax(self.b)
            s = torch.matmul(c.t(), pred)
            v = squash(s)
            self.b = torch.add(self.b, torch.matmul(pred, v.t()))
        return v


if __name__ == '__main__':
    r = Routing(32 * 6 * 6, 10, 8, 16, 32)
    t = Variable(torch.rand(3, 256, 6, 6))
    for _ in range(10):
        r(t)
