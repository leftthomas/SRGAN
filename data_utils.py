from os import listdir
from os.path import join

from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, Scale, RandomCrop, ToTensor, ToPILImage


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def hr_transform(crop_size):
    return Compose([
        RandomCrop(crop_size),
        ToTensor(),
    ])


def lr_transform(crop_size, upscale_factor):
    return Compose([
        ToPILImage(),
        Scale(crop_size // upscale_factor, interpolation=Image.BICUBIC),
        ToTensor()
    ])


class DatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(DatasetFromFolder, self).__init__()
        self.image_dir = dataset_dir
        self.image_filenames = [join(self.image_dir, x) for x in listdir(self.image_dir) if is_image_file(x)]
        self.hr_transform = hr_transform(72)
        self.lr_transform = lr_transform(72, upscale_factor)

    def __getitem__(self, index):
        hr_image = self.hr_transform(Image.open(self.image_filenames[index]))
        lr_image = self.lr_transform(hr_image)
        return lr_image, hr_image

    def __len__(self):
        return len(self.image_filenames)


class AvgMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class LRTransformTest(object):
    def __init__(self, shrink_factor):
        self.to_pil = ToPILImage()
        self.to_tensor = ToTensor()
        self.shrink_factor = shrink_factor

    def __call__(self, hr_tensor):
        hr_img = self.to_pil(hr_tensor)
        w, h = hr_img.size
        lr_scale = Scale(int(min(w, h) / self.shrink_factor), interpolation=3)
        hr_scale = Scale(min(w, h), interpolation=3)
        lr_img = lr_scale(hr_img)
        hr_restore_img = hr_scale(lr_img)
        return self.to_tensor(lr_img), self.to_tensor(hr_restore_img)
