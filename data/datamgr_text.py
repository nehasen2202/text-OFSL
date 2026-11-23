"""Episodic image/label/path loader; original data managers remain untouched."""
import torch
from data.dataset import EpisodicBatchSampler
from data.dataset_text import SetDatasetText
from data.datamgr import TransformLoader


def collate_episode(batch):
    images = torch.stack([item[0] for item in batch], dim=0)
    labels = torch.stack([item[1] for item in batch], dim=0)
    paths = [list(item[2]) for item in batch]
    return images, labels, paths


class SetDataManagerText:
    def __init__(self, image_size, n_way, n_support, n_query, n_eposide=100, num_workers=4):
        self.image_size = image_size
        self.n_way = n_way
        self.batch_size = n_support + n_query
        self.n_eposide = n_eposide
        self.num_workers = num_workers

    def get_data_loader(self, data_file, aug):
        transform = TransformLoader(self.image_size).get_composed_transform(aug)
        dataset = SetDatasetText(data_file, self.batch_size, transform)
        sampler = EpisodicBatchSampler(len(dataset), self.n_way, self.n_eposide)
        return torch.utils.data.DataLoader(dataset, batch_sampler=sampler,
                                           num_workers=self.num_workers, pin_memory=True,
                                           collate_fn=collate_episode)


def get_train_loader_text(image_size, base_file, val_file, train_n_way=2,
                          test_n_way=2, n_shot=5, num_workers=4):
    n_query = max(1, int(16 * test_n_way / train_n_way))
    train = SetDataManagerText(image_size, train_n_way, n_shot, n_query,
                               num_workers=num_workers).get_data_loader(base_file, aug=True)
    val = SetDataManagerText(image_size, test_n_way, n_shot, n_query,
                             num_workers=num_workers).get_data_loader(val_file, aug=False)
    return train, val
