"""Episodic datasets returning (image, class_label, image_path)."""
import json
import numpy as np
import torch
from PIL import Image
from data.dataset import EpisodicBatchSampler


class SubDatasetText(torch.utils.data.Dataset):
    def __init__(self, paths, label, transform):
        self.paths, self.label, self.transform = list(paths), int(label), transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = str(self.paths[idx])
        with Image.open(path) as image:
            x = self.transform(image.convert('RGB'))
        return x, self.label, path


class SetDatasetText(torch.utils.data.Dataset):
    def __init__(self, json_file, batch_size, transform):
        with open(json_file, encoding='utf-8') as f:
            meta = json.load(f)
        labels = np.asarray(meta['image_labels'])
        paths = meta['image_names']
        self.classes = np.unique(labels).tolist()
        self.loaders = []
        for cls in self.classes:
            cls_paths = [p for p, y in zip(paths, labels) if y == cls]
            if len(cls_paths) < batch_size:
                raise ValueError(f'Class {cls} has only {len(cls_paths)} images; need {batch_size}')
            ds = SubDatasetText(cls_paths, cls, transform)
            self.loaders.append(torch.utils.data.DataLoader(ds, batch_size=batch_size,
                                                             shuffle=True, num_workers=0))

    def __len__(self):
        return len(self.classes)

    def __getitem__(self, idx):
        return next(iter(self.loaders[idx]))
