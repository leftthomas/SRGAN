import torch
from torch import nn
from torchvision.models.vgg import vgg16


class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        vgg = vgg16(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:16]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, out_labels, out_images, target_images):
        # Adversarial Loss
        adversarial_loss = -torch.mean(torch.log(out_labels))
        # Perception Loss
        perception_loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))
        # Image Loss
        image_loss = self.l1_loss(out_images, target_images)

        return 170 * image_loss + adversarial_loss + 145 * perception_loss


if __name__ == "__main__":
    g_loss = GeneratorLoss()
    print(g_loss)
