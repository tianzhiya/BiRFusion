import os
import random
import torch
import torch.utils.data as data
from os import listdir
from os.path import join
from torch.utils.data.dataset import Dataset
import numpy as np
from PIL import Image
import cv2
import glob


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".bmp", ".JPG", ".jpeg"])


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img


class DatasetFromFolder(data.Dataset):
    def __init__(self, data_dir, transform=None):
        super(DatasetFromFolder, self).__init__()
        self.data_dir = data_dir
        self.transform = transform

    def __getitem__(self, index):
        index = index
        data_filenames = [join(join(self.data_dir, str(index + 1)), x) for x in
                          listdir(join(self.data_dir, str(index + 1))) if is_image_file(x)]
        num = len(data_filenames)
        index1 = random.randint(1, num)
        index2 = random.randint(1, num)
        while abs(index1 - index2) == 0:
            index2 = random.randint(1, num)

        im1 = load_img(data_filenames[index1 - 1])
        im2 = load_img(data_filenames[index2 - 1])

        _, file1 = os.path.split(data_filenames[index1 - 1])
        _, file2 = os.path.split(data_filenames[index2 - 1])

        seed = np.random.randint(123456789)
        if self.transform:
            random.seed(seed)
            torch.manual_seed(seed)
            im1 = self.transform(im1)
            random.seed(seed)
            torch.manual_seed(seed)
            im2 = self.transform(im2)
        return im1, im2, file1, file2

    def __len__(self):
        return 324


class DatasetFromFolderEval(data.Dataset):
    def __init__(self, data_dir, transform=None):
        super(DatasetFromFolderEval, self).__init__()
        data_filenames = [join(data_dir, x) for x in listdir(data_dir) if is_image_file(x)]
        data_filenames.sort()
        self.data_filenames = data_filenames
        self.transform = transform

    def __getitem__(self, index):
        input = load_img(self.data_filenames[index])
        _, file = os.path.split(self.data_filenames[index])

        if self.transform:
            input = self.transform(input)
        return input, file

    def __len__(self):
        return len(self.data_filenames)


def prepare_data_path(dataset_path):
    filenames = os.listdir(dataset_path)
    data_dir = dataset_path
    data = glob.glob(os.path.join(data_dir, "*.bmp"))
    data.extend(glob.glob(os.path.join(data_dir, "*.tif")))
    data.extend(glob.glob((os.path.join(data_dir, "*.jpg"))))
    data.extend(glob.glob((os.path.join(data_dir, "*.png"))))
    data.sort()
    filenames.sort()
    return data, filenames


class Fusion_dataset(Dataset):
    def __init__(self, split, ir_path=None, vi_path=None):
        super(Fusion_dataset, self).__init__()
        assert split in ['train', 'val', 'test'], 'split must be "train"|"val"|"test"'

        if split == 'train':
            data_dir_vis = './dataset/MSRS/Visible/train/MSRS/'
            data_dir_ir = './dataset/MSRS/Infrared/train/MSRS/'
            data_dir_label = './dataset/MSRS/Label/train/MSRS/'
            self.filepath_vis, self.filenames_vis = prepare_data_path(data_dir_vis)
            self.filepath_ir, self.filenames_ir = prepare_data_path(data_dir_ir)
            self.filepath_label, self.filenames_label = prepare_data_path(data_dir_label)
            self.split = split
            self.length = min(len(self.filenames_vis), len(self.filenames_ir))

        elif split == 'val':
            data_dir_vis = './dataset/MSRS/Visible/test/MSRS/'
            data_dir_ir = './dataset/MSRS/Infrared/test/MSRS/'
            self.filepath_vis, self.filenames_vis = prepare_data_path(data_dir_vis)
            self.filepath_ir, self.filenames_ir = prepare_data_path(data_dir_ir)
            self.split = split
            self.length = min(len(self.filenames_vis), len(self.filenames_ir))

    def __getitem__(self, index):
        if self.split == 'train':
            vis_path = self.filepath_vis[index]
            ir_path = self.filepath_ir[index]
            label_path = self.filepath_label[index]
            image_vis = np.array(Image.open(vis_path))
            image_inf = cv2.imread(ir_path, 0)
            label = np.array(Image.open(label_path))
            image_vis = (
                    np.asarray(Image.fromarray(image_vis), dtype=np.float32).transpose(
                        (2, 0, 1)
                    )
                    / 255.0
            )
            image_ir = np.asarray(Image.fromarray(image_inf), dtype=np.float32) / 255.0
            image_ir = np.expand_dims(image_ir, axis=0)
            name = self.filenames_vis[index]
            return (
                torch.tensor(image_vis),
                torch.tensor(image_ir),
                name, name,
            )
        elif self.split == 'val':
            vis_path = self.filepath_vis[index]
            ir_path = self.filepath_ir[index]
            image_vis = np.array(Image.open(vis_path))
            image_inf = cv2.imread(ir_path, 0)
            image_vis = (
                    np.asarray(Image.fromarray(image_vis), dtype=np.float32).transpose(
                        (2, 0, 1)
                    )
                    / 255.0
            )
            image_ir = np.asarray(Image.fromarray(image_inf), dtype=np.float32) / 255.0
            image_ir = np.expand_dims(image_ir, axis=0)
            name = self.filenames_vis[index]
            return (
                torch.tensor(image_vis),
                torch.tensor(image_ir),
                name, name
            )

    def __len__(self):
        return self.length
