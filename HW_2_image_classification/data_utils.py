import os
from tqdm import tqdm
import numpy as np
import math

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image

from albumentations.core.composition import Compose



class IntelImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None,
                 mode='train', load_on_fly=True, reduced_num=None):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.load_on_fly = load_on_fly
        self.reduced_num = reduced_num

        self.classes = [c for c in os.listdir(self.img_dir) if not c.startswith('.')]
        self.file_paths = []
        self.img_labels = []
        for c in tqdm(self.classes, total=len(self.classes), mininterval=10):
            folder = os.path.join(self.img_dir, c)
            image_names = os.listdir(folder)
            image_paths = [os.path.join(self.img_dir, c, n) for n in image_names]
            if self.reduced_num:
                image_paths = image_paths[: self.reduced_num]
            if not load_on_fly:
                image_paths = [read_image(path) for path in image_paths]  # returns torch.ByteTensor

            if mode == 'train':
                image_paths = image_paths[: int(0.8 * len(image_paths))]
            elif mode == 'val':
                image_paths = image_paths[int(0.8 * len(image_paths)):]
            self.file_paths.extend(image_paths)
            self.img_labels.extend([c] * len(image_paths))

        if mode == 'train' and self.target_transform:
            self.target_transform.fit(np.array(self.img_labels))

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        if self.load_on_fly:
            img_path = self.file_paths[idx]
            # image = pil_loader(img_path)
            image = read_image(img_path)
        else:
            image = self.file_paths[idx]

        label = self.img_labels[idx]
        if self.transform:
            if isinstance(self.transform, Compose):
                image = self.transform(image=image.numpy())
            else:
                image = self.transform(image)
        if self.target_transform:
            label = torch.tensor(self.target_transform.transform([label]))[0]
        return image, label

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            iter_start = self.start
            iter_end = self.end
        else:  # in a worker process
            # split workload
            per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = self.start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end)
        return iter(range(iter_start, iter_end))
