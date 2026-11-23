#!/usr/bin/env python3
"""Generate captions using the original, unfine-tuned microsoft/git-base model.

Usage:
 python generate_git_captions.py --dataset ND0 --image-root /path/to/ND0 --output-dir ./captions
"""
import argparse
import csv
import json
import os
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForCausalLM

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
MODEL_ID = 'microsoft/git-base'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, choices=['miniImagenet'])
    parser.add_argument('--image-root', required=True, type=Path, help='Folder containing this dataset images, recursively')
    parser.add_argument('--output-dir', default=Path('./captions'), type=Path)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--max-length', type=int, default=50)
    parser.add_argument('--revision', default='main', help='Hugging Face model revision; pin commit for final reproducibility')
    args = parser.parse_args()
    root = args.image_root.expanduser().resolve()
    if not root.is_dir():
        parser.error(f'Image root does not exist: {root}')
    if args.batch_size < 1 or args.max_length < 1:
        parser.error('batch-size and max-length must be positive')
    paths = sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if not paths:
        parser.error(f'No supported images under {root}')
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / f'{args.dataset}_pretrained_GIT_captions.csv'
    errors_path = out / f'{args.dataset}_pretrained_GIT_errors.log'
    config_path = out / f'{args.dataset}_pretrained_GIT_config.json'
    config = {'dataset': args.dataset, 'image_root': str(root), 'model': MODEL_ID,
              'revision': args.revision, 'batch_size': args.batch_size,
              'max_length': args.max_length, 'num_beams': 1, 'do_sample': False}
    # Prevent accidental mixing of captions from different models or generation settings.
    if config_path.exists():
        previous = json.loads(config_path.read_text(encoding='utf-8'))
        if previous != config:
            parser.error(f'Existing configuration differs: {config_path}. Use another output-dir or restore original settings.')
    else:
        if csv_path.exists():
            parser.error(f'CSV exists without config: {csv_path}. Use a new output-dir or inspect it first.')
        config_path.write_text(json.dumps(config, indent=2), encoding='utf-8')
    completed = set()
    if csv_path.exists():
        with csv_path.open(newline='', encoding='utf-8') as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != ['image_path', 'caption']:
                parser.error('Existing CSV has unexpected columns')
            completed = {row['image_path'] for row in reader if row.get('caption', '').strip()}
    remaining = [p for p in paths if str(p) not in completed]
    print(f'{args.dataset}: {len(paths)} images, {len(completed)} completed, {len(remaining)} remaining')
    if not remaining:
        return
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Loading {MODEL_ID} on {device}')
    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=args.revision).to(device).eval()

    def caption_batch(batch, writer, err):
        images, valid = [], []
        for p in batch:
            try:
                with Image.open(p) as im:
                    images.append(im.convert('RGB'))
                valid.append(p)
            except Exception as exc:
                err.write(f'{p}\t{type(exc).__name__}: {exc}\n')
        if not valid:
            return
        try:
            inputs = processor(images=images, return_tensors='pt').to(device)
            with torch.inference_mode():
                ids = model.generate(pixel_values=inputs['pixel_values'], max_length=args.max_length,
                                     num_beams=1, do_sample=False)
            captions = processor.batch_decode(ids, skip_special_tokens=True)
            if len(captions) != len(valid):
                raise RuntimeError('Number of captions does not match number of images')
            for p, cap in zip(valid, captions):
                if not cap.strip():
                    err.write(f'{p}\tEmpty caption\n')
                else:
                    writer.writerow([str(p), cap.strip()])
        except Exception as exc:
            if len(valid) > 1:
                for p in valid:
                    caption_batch([p], writer, err)
            else:
                err.write(f'{valid[0]}\t{type(exc).__name__}: {exc}\n')

    has_header = csv_path.exists() and csv_path.stat().st_size > 0
    with csv_path.open('a', newline='', encoding='utf-8') as handle, errors_path.open('a', encoding='utf-8') as err:
        writer = csv.writer(handle)
        if not has_header:
            writer.writerow(['image_path', 'caption'])
        for i in tqdm(range(0, len(remaining), args.batch_size), desc=args.dataset):
            caption_batch(remaining[i:i + args.batch_size], writer, err)
            handle.flush()
            err.flush()
    print(f'Finished. Captions: {csv_path}\nErrors: {errors_path}')


if __name__ == '__main__':
    main()

