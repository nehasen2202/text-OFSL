"""IDEAL + frozen BERT caption embeddings, additive prototype-score fusion.

Keeps original IDEAL visual calibration and losses unchanged. The text branch
uses per-class mean prototypes and cosine similarities (10x logit scale).
"""
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from methods.IDEAL import IDEAL
from methods.noisemeta_template_text import NoiseMetaTemplateText
from sklearn.metrics import precision_score, recall_score, f1_score


def canonical_path(p):
    return str(Path(str(p)).expanduser().resolve(strict=False))


class BertFeatureStore:
    def __init__(self, files):
        from torch.torch_version import TorchVersion

        torch.serialization.add_safe_globals([TorchVersion])

        self.vectors = {}

        for filename in files:
            data = torch.load(
                filename,
                map_location='cpu',
                weights_only=True
            )

            paths, vectors = data['image_paths'], data['embeddings']
            if len(paths) != len(vectors) or vectors.ndim != 2 or vectors.shape[1] != 768:
                raise ValueError(f'Invalid BERT features in {filename}')
            for path, vector in zip(paths, vectors):
                key = canonical_path(path)
                if key in self.vectors:
                    if not torch.allclose(self.vectors[key], vector.float(), atol=1e-5):
                        raise ValueError(f'Conflicting embeddings for {key}')
                else:
                    self.vectors[key] = vector.float().clone()
        print(f'Loaded {len(self.vectors)} unique BERT image embeddings')

    def verify_json(self, filename):
        import json
        with open(filename, encoding='utf-8') as f:
            meta = json.load(f)
        missing = [p for p in meta['image_names'] if canonical_path(p) not in self.vectors]
        if missing:
            raise ValueError(f'{len(missing)} missing embeddings in {filename}; first: {missing[0]}')
        print(f'BERT coverage OK: {filename} ({len(meta["image_names"])} images)')

    def get(self, paths, device):
        try:
            vectors = [self.vectors[canonical_path(p)] for row in paths for p in row]
        except KeyError as e:
            raise KeyError(f'BERT embedding not found for image: {e}. Check exact paths in the .pt files.') from e
        return torch.stack(vectors).to(device, non_blocking=True).reshape(len(paths), len(paths[0]), 768)


class IDEALText(IDEAL, NoiseMetaTemplateText):
    def __init__(self, *args, feature_store, text_weight=0.3, **kwargs):
        super().__init__(*args, **kwargs)
        self.feature_store = feature_store
        self.text_weight = float(text_weight)
        if self.text_weight < 0:
            raise ValueError('text_weight must be >= 0')
        self._active_paths = None

    def set_forward(self, x):
        visual_scores = super().set_forward(x)
        if self._active_paths is None:
            raise RuntimeError('Missing image paths for BERT fusion')
        text = self.feature_store.get(self._active_paths, self.device)
        text_support = text[:, :self.n_support]
        text_query = text[:, self.n_support:]
        # Unweighted text prototypes: preserves IDEAL visual calibration as-is.
        prototypes = text_support.mean(dim=1)
        query = text_query.reshape(-1, 768)
        text_scores = 10.0 * (F.normalize(query, dim=1) @
                              F.normalize(prototypes, dim=1).T)
        return visual_scores + self.text_weight * text_scores

    def train_loop(self, epoch, train_loader, optimizer, image_files=None, image_labels=None):
        self.train()
        if image_files is not None:
            self.image_files = np.asarray(image_files)
            self.image_labels = np.asarray(image_labels)
        total = 0.0
        count = 0
        for x, y, paths in train_loader:
            self.n_way = x.size(0)
            self.n_query = x.size(1) - self.n_support
            if self.image_size is None:
                self.image_size = x.shape[-1]
            x, y, indices, updated_paths = self.add_noise_with_paths(
                x, y, paths, self.noise_type, aug=True, return_y=True)
            self._active_paths = updated_paths
            x, y = x.to(self.device), y.to(self.device)
            optimizer.zero_grad()
            loss = self.set_forward_loss(x, y, indices)
            loss.backward()
            optimizer.step()
            total += loss.item()
            count += 1
        self._active_paths = None
        print(f'Epoch {epoch} | Loss {total / max(count, 1):.6f}')
        return total

    def test_loop(self, test_loader, return_std=False, image_files=None,
                  image_labels=None, return_metrics=False):
        self.eval()
        if image_files is not None:
            self.image_files = np.asarray(image_files)
            self.image_labels = np.asarray(image_labels)
        acc, precision, recall, f1 = [], [], [], []
        for x, y, paths in test_loader:
            self.n_way = x.size(0)
            self.n_query = x.size(1) - self.n_support
            if self.image_size is None:
                self.image_size = x.shape[-1]
            x, updated_paths = self.add_noise_with_paths(x, y, paths, self.noise_type, aug=False)
            self._active_paths = updated_paths
            with torch.no_grad():
                scores = self.set_forward(x.to(self.device))
            target = np.repeat(np.arange(self.n_way), self.n_query)
            pred = scores.argmax(dim=1).cpu().numpy()
            labels = np.arange(self.n_way)
            acc.append(float((pred == target).mean() * 100))
            precision.append(precision_score(target, pred, labels=labels, average='macro', zero_division=0) * 100)
            recall.append(recall_score(target, pred, labels=labels, average='macro', zero_division=0) * 100)
            f1.append(f1_score(target, pred, labels=labels, average='macro', zero_division=0) * 100)
        self._active_paths = None
        if not acc:
            raise ValueError('No evaluation episodes')
        mean = float(np.mean(acc))
        std = float(np.std(acc))
        ci = float(1.96 * std / np.sqrt(len(acc)))
        if return_metrics:
            return dict(accuracy=mean, accuracy_std=std, accuracy_ci95=ci,
                        precision_macro=float(np.mean(precision)), recall_macro=float(np.mean(recall)),
                        f1_macro=float(np.mean(f1)), episodes=len(acc))
        return (mean, std) if return_std else mean

