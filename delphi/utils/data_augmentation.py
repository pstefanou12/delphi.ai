# Dataset augmentation info

from torch import Tensor
from torchvision import transforms

# CIFAR10 dataset info
CIFAR_LABELS = {
    0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer',
    5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck',
}


# CIFAR10 dataset information
_CIFAR10_STATS = {'mean': [0.4914, 0.4822, 0.4465],
                  'std': [0.2023, 0.1994, 0.2010]}

# IMAGENET dataset information
_IMAGENET_STATS = {'mean': [0.485, 0.456, 0.406],
                   'std': [0.229, 0.224, 0.225]}

_IMAGENET_PCA = {
    'eigval': Tensor([0.2175, 0.0188, 0.0045]),
    'eigvec': Tensor([
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ])
}

# Dataset training set augmentation defaults
TRAIN_TRANSFORMS_DEFAULT = lambda size: transforms.Compose([
        transforms.RandomCrop(size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(.25, .25, .25),
        transforms.RandomRotation(2),
        transforms.ToTensor(),
    ])
"""
Generic training data transform, given image side length does random cropping,
flipping, color jitter, and rotation. Called as, for example,
:meth:`delphi.data_augmentation.TRAIN_TRANSFORMS_DEFAULT(32)` for CIFAR-10.
"""

# Dataset test set augmentation defaults
TEST_TRANSFORMS_DEFAULT = lambda size: transforms.Compose([
    transforms.Resize(size),
    transforms.CenterCrop(size),
    transforms.ToTensor()
])
"""
Generic test data transform (no augmentation) to complement
:meth:`delphi.data_augmentation.TEST_TRANSFORMS_DEFAULT`, takes in an image
side length.
"""


class Lighting(object):
    """
    Lighting noise (see https://git.io/fhBOc)
    """
    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone()\
            .mul(alpha.view(1, 3).expand(3, 3))\
            .mul(self.eigval.view(1, 3).expand(3, 3))\
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))
