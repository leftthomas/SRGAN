import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models.vgg import vgg16


class CapsuleLoss(nn.Module):
    def __init__(self):
        super(CapsuleLoss, self).__init__()
        self.reconstruction_loss = nn.MSELoss(size_average=False)

    def forward(self, images, labels, classes, reconstructions):
        left = F.relu(0.9 - classes, inplace=True) ** 2
        right = F.relu(classes - 0.1, inplace=True) ** 2

        margin_loss = labels * left + 0.5 * (1. - labels) * right
        margin_loss = margin_loss.sum()

        reconstruction_loss = self.reconstruction_loss(reconstructions, images)

        return (margin_loss + 0.0005 * reconstruction_loss) / images.size(0)


def vgg16_relu2_2():
    vgg = vgg16(pretrained=True)
    relu2_2 = nn.Sequential(*list(vgg.features)[:9])
    for param in relu2_2.parameters():
        param.requires_grad = False
    relu2_2.eval()
    return relu2_2


class ContentLoss(nn.Module):
    def __init__(self, loss_network):
        super(ContentLoss, self).__init__()
        self.loss_network = loss_network

    def forward(self, input, target):
        features_input = self.loss_network(input)
        features_target = self.loss_network(target)
        return torch.mean((features_input - features_target) ** 2) + 3 * F.mse_loss(input, target)


if __name__ == "__main__":
    digit_loss = CapsuleLoss()
    print(digit_loss)
