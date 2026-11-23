"""Noise injection with reliable image-path tracking for multimodal episodes."""
import numpy as np
from methods.noisemeta_template import NoiseMetaTemplate


class NoiseMetaTemplateText(NoiseMetaTemplate):
    def add_noise_with_paths(self, x, y, paths, noise_type, aug, return_y=False):
        # DataLoader collates strings as a nested class-major sequence.
        if len(paths) != x.shape[0] or any(len(row) != x.shape[1] for row in paths):
            raise ValueError('Image path dimensions do not match the episode tensor')
        original_paths = [list(map(str, row)) for row in paths]
        result = super().add_noise(x, y, noise_type=noise_type, aug=aug, return_y=return_y)
        metadata = self.last_noise_metadata or {}
        updated = [row.copy() for row in original_paths]
        for rep in metadata.get('replacements', []):
            i, j = rep['target_class_index'], rep['target_support_index']
            if rep['source_path'] is not None:
                updated[i][j] = str(rep['source_path'])
            else:
                si, sj = rep['source_class_index'], rep['source_support_index']
                updated[i][j] = original_paths[si][sj]
        if return_y:
            x_new, y_new, indices = result
            return x_new, y_new, indices, updated
        return result, updated
