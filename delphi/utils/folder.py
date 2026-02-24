"""
Image folder dataset utilities with support for label mappings and custom loaders.
"""

import os
import os.path
import sys

from torch.utils import data
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision import get_image_backend

try:
    import accimage
except ImportError:
    accimage = None
from PIL import Image


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
        extensions (iterable of strings): extensions to consider (lowercase)
    Returns:
        bool: True if the filename ends with one of given extensions
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def is_image_file(filename):
    """Checks if a file is an allowed image extension.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def make_dataset(directory, class_to_idx, extensions):
    """Build a list of (path, class_index) tuples from a root directory."""
    images = []
    directory = os.path.expanduser(directory)
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(directory, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images


class DatasetFolder(data.Dataset):  # pylint: disable=too-many-instance-attributes
    """A generic data loader where the samples are arranged in this way: ::
        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext
        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext
    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (list[string]): A list of allowed extensions.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        root,
        loader,
        extensions,
        transform=None,
        target_transform=None,
        label_mapping=None,
    ):
        """Initialize DatasetFolder, finding classes and building sample list."""
        classes, class_to_idx = self._find_classes(root)
        if label_mapping is not None:
            classes, class_to_idx = label_mapping(classes, class_to_idx)

        samples = make_dataset(root, class_to_idx, extensions)
        if len(samples) == 0:
            raise RuntimeError(
                f"Found 0 files in subfolders of: {root}\n"
                f"Supported extensions are: {','.join(extensions)}"
            )

        self.root = root
        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

        self.transform = transform
        self.target_transform = target_transform

    def _find_classes(self, directory):
        """
        Finds the class folders in a dataset.
        Args:
            directory (string): Root directory path.
        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (directory),
            and class_to_idx is a dictionary.
        Ensures:
            No class is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(directory) if d.is_dir()]
        else:
            classes = [
                d
                for d in os.listdir(directory)
                if os.path.isdir(os.path.join(directory, d))
            ]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = f"Dataset {self.__class__.__name__}\n"
        fmt_str += f"    Number of datapoints: {self.__len__()}\n"
        fmt_str += f"    Root Location: {self.root}\n"
        tmp = "    Transforms (if any): "
        nl_indent = chr(10) + " " * len(tmp)
        fmt_str += f"{tmp}{self.transform.__repr__().replace(chr(10), nl_indent)}\n"
        tmp = "    Target Transforms (if any): "
        nl_indent2 = chr(10) + " " * len(tmp)
        fmt_str += (
            f"{tmp}{self.target_transform.__repr__().replace(chr(10), nl_indent2)}"
        )
        return fmt_str


IMG_EXTENSIONS = [".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif"]


def pil_loader(path):
    """Load an image from path using PIL."""
    # open path as file to avoid ResourceWarning
    # https://github.com/python-pillow/Pillow/issues/835
    with open(path, "rb") as f:  # pylint: disable=unspecified-encoding
        img = Image.open(f)
        return img.convert("RGB")


def accimage_loader(path):
    """Load an image using accimage, falling back to PIL on IOError."""
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    """Load an image using the backend determined by torchvision."""
    if get_image_backend() == "accimage":
        return accimage_loader(path)
    return pil_loader(path)


class ImageFolder(DatasetFolder):  # pylint: disable=too-few-public-methods
    """A generic data loader where the images are arranged in this way: ::
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        root,
        transform=None,
        target_transform=None,
        loader=default_loader,
        label_mapping=None,
    ):
        """Initialize ImageFolder from root directory."""
        super().__init__(
            root,
            loader,
            IMG_EXTENSIONS,
            transform=transform,
            target_transform=target_transform,
            label_mapping=label_mapping,
        )
        self.imgs = self.samples


class TensorDataset(Dataset):
    """Dataset wrapping tensors.
    Each sample will be retrieved by indexing tensors along the first dimension.
    Arguments:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """

    def __init__(self, *tensors, transform=None):
        """Initialize TensorDataset with tensors and optional transform."""
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        """Return (image, target) pair at index, applying transform if set."""
        im, targ = tuple(tensor[index] for tensor in self.tensors)

        if self.transform:
            real_transform = transforms.Compose(
                [transforms.ToPILImage(), self.transform]
            )
            im = real_transform(im)

        return im, targ

    def __len__(self):
        return self.tensors[0].size(0)
