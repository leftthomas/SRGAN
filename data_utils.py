from os import listdir
from os.path import join

from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, Scale, RandomCrop, ToTensor, ToPILImage, CenterCrop


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def train_hr_transform(crop_size):
    return Compose([
        RandomCrop(crop_size),
        ToTensor(),
    ])


def val_hr_transform(crop_size):
    return Compose([
        CenterCrop(crop_size),
        ToTensor(),
    ])


def lr_transform(crop_size, upscale_factor):
    return Compose([
        ToPILImage(),
        Scale(crop_size // upscale_factor, interpolation=Image.BICUBIC),
        ToTensor()
    ])


def upscale_transform(crop_size):
    return Compose([
        ToPILImage(),
        Scale(crop_size, interpolation=Image.BICUBIC),
        ToTensor()
    ])


class DatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, hr_transform, lr_transform):
        super(DatasetFromFolder, self).__init__()
        self.image_dir = dataset_dir
        self.image_filenames = [join(self.image_dir, x) for x in listdir(self.image_dir) if is_image_file(x)]
        self.hr_transform = hr_transform
        self.lr_transform = lr_transform

    def __getitem__(self, index):
        hr_image = self.hr_transform(Image.open(self.image_filenames[index]))
        lr_image = self.lr_transform(hr_image)
        return lr_image, hr_image

    def __len__(self):
        return len(self.image_filenames)

