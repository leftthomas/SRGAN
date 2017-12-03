from torch import nn
from torchvision.models.vgg import vgg16


class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        vgg = vgg16(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()

    def forward(self, out_labels, out_images, target_images):
        # Adversarial Loss
        adversarial_loss = - out_labels.mean()
        # Perception Loss
        perception_loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))
        # Image Loss
        image_loss = self.mse_loss(out_images, target_images)
        # TV Loss
        tv_loss = (((out_images[:, :, :-1, :] - out_images[:, :, 1:, :]) ** 2 + (
            out_images[:, :, :, :-1] - out_images[:, :, :, 1:]) ** 2) ** 1.25).mean()

        return image_loss + 1e-3 * adversarial_loss + 6e-3 * perception_loss + 2e-8 * tv_loss


if __name__ == "__main__":
    g_loss = GeneratorLoss()
    print(g_loss)
