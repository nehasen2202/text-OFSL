# From Instance to Class Calibration: A Unified Framework for Open-World Few-Shot Learning. IEEE TPAMI 2023

import torch
import torch.nn as nn
import numpy as np
from abc import abstractmethod
from torchvision.transforms import Resize, CenterCrop, ToTensor, RandomResizedCrop, RandomHorizontalFlip, Normalize
from PIL import ImageEnhance, Image
import copy


class NoiseMetaTemplate(nn.Module):
    def __init__(self, model_func, n_way, n_support, outlier=1, verbose=False, image_size=None, image_files=None,
                 image_labels=None, noise_type='IT', noise_rate=0.5, device='cuda:0'):
        super(NoiseMetaTemplate, self).__init__()
        self.n_way = n_way  # N, n_classes
        self.n_support = n_support  # S, sample num of support set
        self.n_query = -1  # Q, sample num of query set(change depends on input)
        self.feature_extractor = model_func()  # feature extractor
        self.feat_dim = self.feature_extractor.final_feat_dim
        self.verbose = verbose
        self.noise_type = noise_type
        assert self.noise_type in ['IT', 'OOT', 'OOD']
        self.outlier = outlier
        self.noise_rate = noise_rate
        self.image_files = np.asarray(image_files) if image_files is not None else np.array([])
        self.image_labels = np.asarray(image_labels) if image_labels is not None else np.array([])

        # Optional information about the most recent noise injection.
        # Useful later for IDEAL+Text image-caption alignment.
        self.last_noise_metadata = None
        self.image_size = image_size
        self.device = device

    @abstractmethod
    def set_forward(self, x):
        # x -> predicted score
        pass

    @abstractmethod
    def set_forward_loss(self, x):
        # x -> loss value
        pass

    def forward(self, x):
        # x-> feature embedding
        out = self.feature_extractor.forward(x)
        return out

    def parse_feature(self, x):
        x = x.requires_grad_(True)
        x = x.reshape(self.n_way * (self.n_support + self.n_query), *x.size()[2:])
        z_all = self.feature_extractor.forward(x)
        z_all = z_all.reshape(self.n_way, self.n_support + self.n_query, *z_all.shape[1:])  # [N, S+Q, d]
        z_support = z_all[:, :self.n_support]  # [N, S, d]
        z_query = z_all[:, self.n_support:]  # [N, Q, d]
        return z_support, z_query

    def correct(self, x):
        scores = self.set_forward(x)
        y_query = np.repeat(range(self.n_way), self.n_query)  # [0 0 0 1 1 1 2 2 2 3 3 3 4 4 4]
        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)  # top1, dim=1, largest, sorted
        topk_ind = topk_labels.cpu().numpy()  # index of topk
        top1_correct = np.sum(topk_ind[:, 0] == y_query)
        return float(top1_correct), len(y_query)

    def train_loop(self, epoch, train_loader, optimizer, image_files=None, image_labels=None):
        self.train()
        if image_files is not None:
            self.image_files = image_files
            self.image_labels = image_labels
        print_freq = 10
        avg_loss = 0
        for i, (x, y) in enumerate(train_loader):
            if self.image_size is None: self.image_size = x.shape[-2:]
            x = self.add_noise(x, y, noise_type=self.noise_type, aug=True)
            x = x.to(self.device)
            self.n_query = x.size(1) - self.n_support  # x:[N, S+Q, n_channel, h, w]
            self.n_way = x.size(0)
            optimizer.zero_grad()
            loss = self.set_forward_loss(x)
            loss.backward()
            optimizer.step()
            avg_loss = avg_loss + loss.item()
            if self.verbose and (i % print_freq) == 0:
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader),
                                                                        avg_loss / float(i + 1)))
        if not self.verbose:
            print('Epoch {:d} | Loss {:f}'.format(epoch, avg_loss / float(i + 1)))
        return avg_loss

    def test_loop(self, test_loader, return_std=False, image_files=None, image_labels=None):
        self.eval()
        if image_files is not None:
            self.image_files = image_files
            self.image_labels = image_labels
        acc_all = []
        iter_num = len(test_loader)
        for i, (x, y) in enumerate(test_loader):
            x = self.add_noise(x, y, noise_type=self.noise_type, aug=False)
            x = x.to(self.device)
            self.n_query = x.size(1) - self.n_support  # x:[N, S+Q, n_channel, h, w]
            self.n_way = x.size(0)
            correct_this, count_this = self.correct(x)
            acc_all.append(correct_this / count_this * 100)
        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)
        if self.verbose:
            # Confidence Interval   90% -> 1.645      95% -> 1.96     99% -> 2.576
            print('%d Test Acc = %4.2f±%4.2f' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))
        if return_std:
            return acc_mean, acc_std
        else:
            return acc_mean

    def add_noise(self, x, y, noise_type, aug, return_y=False):

        n_way = x.shape[0]

        if self.outlier == 0:
            self.last_noise_metadata = {
                'noise_type': 'clean',
                'replacements': []
            }
            if return_y:
                empty_indices = np.empty((x.shape[0], 0), dtype=int)
                return x, y.clone().to(self.device).long(), empty_indices
            return x

        if not (0 <= self.outlier <= self.n_support):
            raise ValueError(
                f"Invalid number of noisy samples: {self.outlier}. "
                f"Expected a value between 0 and {self.n_support}."
            )

        # Preserve the class order used by the episodic data loader.
        # Each row of y corresponds to one episode class.
        episode_labels = y[:, 0].detach().cpu().numpy()

        if len(np.unique(episode_labels)) != n_way:
            raise ValueError(
                "Each episode row must correspond to a different class."
            )

        # Preserve actual class labels for IDEAL's inter-class loss.
        y_new = y.clone().to(self.device).long()

        # Select support positions to contaminate.
        noise_idxes = np.array([
            np.random.choice(
                self.n_support,
                size=self.outlier,
                replace=False
            )
            for _ in range(n_way)
        ])

        # Metadata is optional and does not change IDEAL's return values.
        metadata = {
            'noise_type': noise_type,
            'episode_labels': episode_labels.tolist(),
            'replacements': []
        }

        if noise_type == 'IT':

            # Snapshot the original support images and labels.
            # This avoids overwriting an image before it is reused.
            original_x = x.clone()
            original_y = y_new.clone()

            for ii in range(n_way):

                source_class = (ii + 1) % n_way

                for jj in range(self.outlier):

                    target_idx = int(noise_idxes[ii, jj])
                    source_idx = int(noise_idxes[source_class, jj])

                    x[ii, target_idx] = original_x[
                        source_class, source_idx
                    ]

                    y_new[ii, target_idx] = original_y[
                        source_class, source_idx
                    ]

                    metadata['replacements'].append({
                        'target_class_index': ii,
                        'target_support_index': target_idx,
                        'source_class_index': source_class,
                        'source_support_index': source_idx,
                        'source_label': int(
                            original_y[source_class, source_idx].item()
                        ),
                        'source_path': None
                    })

        elif noise_type == 'OOT':

            image_files = np.asarray(self.image_files)
            image_labels = np.asarray(self.image_labels)

            if len(image_files) != len(image_labels):
                raise ValueError(
                    "OOT image files and labels have different lengths."
                )

            # Exclude every class participating in the episode.
            eligible_mask = ~np.isin(
                image_labels,
                episode_labels
            )

            eligible_indices = np.flatnonzero(eligible_mask)

            if len(eligible_indices) == 0:
                raise ValueError(
                    "No eligible OOT images found. "
                    "Check the episode classes and noise pool."
                )

            for ii in range(n_way):

                for jj in range(self.outlier):

                    target_idx = int(noise_idxes[ii, jj])

                    file_idx = int(
                        np.random.choice(eligible_indices)
                    )

                    source_path = str(image_files[file_idx])
                    source_label = int(image_labels[file_idx])

                    with Image.open(source_path) as img:
                        xx = img.convert('RGB')

                    if aug:
                        xx = RandomResizedCrop(self.image_size)(xx)
                        xx = ImageJitter()(xx)
                        xx = RandomHorizontalFlip()(xx)
                        xx = ToTensor()(xx)
                        xx = Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]
                        )(xx)
                    else:
                        xx = Resize([
                            int(self.image_size * 1.15),
                            int(self.image_size * 1.15)
                        ])(xx)
                        xx = CenterCrop(self.image_size)(xx)
                        xx = ToTensor()(xx)
                        xx = Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]
                        )(xx)

                    x[ii, target_idx] = xx

                    # Store the actual class label, not the pool index.
                    y_new[ii, target_idx] = source_label

                    metadata['replacements'].append({
                        'target_class_index': ii,
                        'target_support_index': target_idx,
                        'source_class_index': None,
                        'source_support_index': None,
                        'source_label': source_label,
                        'source_path': source_path
                    })

        elif noise_type == 'OOD':

            if len(self.image_files) == 0:
                raise ValueError(
                    "OOD noise requires an external image pool."
                )

            for ii in range(n_way):

                for jj in range(self.outlier):

                    target_idx = int(noise_idxes[ii, jj])

                    source_path = str(
                        np.random.choice(self.image_files)
                    )

                    with Image.open(source_path) as img:
                        xx = img.convert('RGB')

                    if aug:
                        xx = RandomResizedCrop(self.image_size)(xx)
                        xx = ImageJitter()(xx)
                        xx = RandomHorizontalFlip()(xx)
                        xx = ToTensor()(xx)
                        xx = Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]
                        )(xx)
                    else:
                        xx = Resize([
                            int(self.image_size * 1.15),
                            int(self.image_size * 1.15)
                        ])(xx)
                        xx = CenterCrop(self.image_size)(xx)
                        xx = ToTensor()(xx)
                        xx = Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]
                        )(xx)

                    x[ii, target_idx] = xx

                    # Assign an artificial label distinct from all
                    # seven ND class labels (assumed to be 0 through 6).
                    y_new[ii, target_idx] = -1

                    metadata['replacements'].append({
                        'target_class_index': ii,
                        'target_support_index': target_idx,
                        'source_class_index': None,
                        'source_support_index': None,
                        'source_label': -1,
                        'source_path': source_path
                    })

        else:
            raise ValueError(
                f"Unsupported noise type: {noise_type}"
            )

        self.last_noise_metadata = metadata

        if return_y:
            return x, y_new, noise_idxes
        else:
            return x

    def cosine_similarity(self, x, y):
        '''
        Cosine Similarity of two tensors
        Args:
            x: torch.Tensor, m x d
            y: torch.Tensor, n x d
        Returns:
            result, m x n
        '''
        assert x.size(1) == y.size(1)
        x = torch.nn.functional.normalize(x, dim=1)
        y = torch.nn.functional.normalize(y, dim=1)
        return x @ y.transpose(0, 1)

    def mahalanobis_dist(self, x, y):
        # x: m x d
        # y: n x d
        # return: m x n
        assert x.size(1) == y.size(1)
        cov = torch.cov(x)  # [m,m]
        x = x.unsqueeze(1).expand(x.size(0), y.size(0), x.size(1))  # [m,1*n,d]
        y = y.unsqueeze(0).expand(x.shape)  # [1*m,n,d]
        delta = x - y  # [m,n,d]
        return torch.einsum('abc,abc->ab', torch.einsum('abc,ad->abc', delta, torch.inverse(cov)), delta)

    def euclidean_dist(self, x, y):
        # x: m x d
        # y: n x d
        # return: m x n
        assert x.size(1) == y.size(1)
        x = x.unsqueeze(1).expand(x.size(0), y.size(0), x.size(1))  # [m,1*n,d]
        y = y.unsqueeze(0).expand(x.shape)  # [1*m,n,d]
        return torch.pow(x - y, 2).sum(2)


class ImageJitter(object):
    def __init__(self, transformtypedict=dict(Brightness=ImageEnhance.Brightness, Contrast=ImageEnhance.Contrast,
                                              Sharpness=ImageEnhance.Sharpness, Color=ImageEnhance.Color),
                 transformdict=dict(Brightness=0.4, Contrast=0.4, Color=0.4)):
        self.transforms = [(transformtypedict[k], transformdict[k]) for k in transformdict]

    def __call__(self, img):
        out = img
        randtensor = torch.rand(len(self.transforms))

        for i, (transformer, alpha) in enumerate(self.transforms):
            r = alpha * (randtensor[i] * 2.0 - 1.0) + 1
            out = transformer(out).enhance(r).convert('RGB')

        return out
