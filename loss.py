import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models.vgg import vgg19


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


def vgg19_relu4_4():
    vgg = vgg19(pretrained=True)
    relu4_4 = nn.Sequential(*list(vgg.features)[:27])
    for param in relu4_4.parameters():
        param.requires_grad = False
    relu4_4.eval()
    return relu4_4


class GeneratorLoss(nn.Module):
    def __init__(self, loss_network):
        super(GeneratorLoss, self).__init__()
        self.loss_network = loss_network
        self.adversarial_loss = nn.BCELoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, out_images, target_images, out_labels, target_labels):
        # Content Loss
        features_input = self.loss_network(out_images)
        features_target = self.loss_network(target_images)
        content_loss = torch.mean((features_input - features_target) ** 2)
        # Adversarial Loss
        adversarial_loss = self.adversarial_loss(out_labels, target_labels)
        # L1 Loss
        l1_loss = self.l1_loss(out_images, target_images)
        return 145 * content_loss + 170 * l1_loss + adversarial_loss


if __name__ == "__main__":
    digit_loss = CapsuleLoss()
    print(digit_loss)
