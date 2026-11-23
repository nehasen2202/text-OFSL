#!/usr/bin/env python3
"""Encode pretrained GIT caption CSVs with frozen bert-base-uncased.

Example:
 python extract_bert_embeddings.py --dataset ND0 --captions ./captions/ND0_pretrained_GIT_captions.csv --output-dir ./bert_features

Outputs a .pt dictionary (image_paths, embeddings, captions, metadata) and metadata JSON.
The row order is explicitly defined by image_paths; downstream code MUST join by path.
"""
import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset', required=True, choices=['miniImagenet'])
    ap.add_argument('--captions', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, default=Path('./bert_features'))
    ap.add_argument('--model-id', default='bert-base-uncased')
    ap.add_argument('--revision', default='main')
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--max-length', type=int, default=64)
    ap.add_argument('--allow-incomplete', action='store_true', help='Allow caption count different from expected dataset size')
    args = ap.parse_args()
    if args.batch_size < 1 or args.max_length < 2:
        ap.error('batch-size must be >= 1 and max-length >= 2')
    if not args.captions.is_file():
        ap.error(f'Caption CSV not found: {args.captions}')

    rows = []
    seen = set()
    with args.captions.open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        if not {'image_path', 'caption'}.issubset(reader.fieldnames or []):
            ap.error('CSV must contain image_path and caption columns')
        for line_num, row in enumerate(reader, start=2):
            path = (row.get('image_path') or '').strip()
            cap = (row.get('caption') or '').strip()
            if not path or not cap:
                ap.error(f'Empty image_path/caption at CSV line {line_num}')
            if path in seen:
                ap.error(f'Duplicate image_path at CSV line {line_num}: {path}')
            seen.add(path)
            rows.append((path, cap))

    expected = 7000 if args.dataset == 'ND3' else 4800
    if len(rows) != expected and not args.allow_incomplete:
        ap.error(f'Found {len(rows)} captions, expected {expected} for {args.dataset}. '
                 'Complete caption generation first, or use --allow-incomplete for testing.')
    if not rows:
        ap.error('Caption CSV contains no rows')
    # Stable ordering independent of CSV append/resume order.
    rows.sort(key=lambda r: r[0])
    image_paths = [r[0] for r in rows]
    captions = [r[1] for r in rows]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'{args.dataset}: {len(rows)} captions, device={device}')
    print(f'Loading frozen {args.model_id} ...')
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, revision=args.revision)
    model = AutoModel.from_pretrained(args.model_id, revision=args.revision).to(device)
    model.eval()
    features = []
    for start in tqdm(range(0, len(captions), args.batch_size), desc='Encoding captions'):
        batch = captions[start:start + args.batch_size]
        tokens = tokenizer(batch, padding=True, truncation=True,
                           max_length=args.max_length, return_tensors='pt')
        tokens = {k: v.to(device) for k, v in tokens.items()}
        with torch.inference_mode():
            output = model(**tokens)
            hidden = output.last_hidden_state
            mask = tokens['attention_mask'].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        features.append(pooled.cpu().float())
    embeddings = torch.cat(features, dim=0)
    if embeddings.shape != (len(rows), 768):
        raise RuntimeError(f'Unexpected embedding shape: {embeddings.shape}')
    if not torch.isfinite(embeddings).all():
        raise RuntimeError('Nonfinite embedding values found')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f'{args.dataset}_BERT_embeddings.pt'
    meta_path = args.output_dir / f'{args.dataset}_BERT_config.json'
    if output_path.exists() or meta_path.exists():
        ap.error(f'Output exists in {args.output_dir}. Move or rename it before regenerating.')
    metadata = {
        'dataset': args.dataset, 'caption_csv': str(args.captions.resolve()),
        'model_id': args.model_id, 'revision': args.revision,
        'pooling': 'attention-mask mean of last_hidden_state (including special tokens)',
        'max_length': args.max_length, 'batch_size': args.batch_size,
        'frozen': True, 'normalize_embeddings': False,
        'num_images': len(rows), 'embedding_dim': 768,
        'alignment': 'match by exact image_path; do not assume JSON or CSV row order',
        'torch_version': torch.__version__,
    }
    torch.save({'image_paths': image_paths, 'captions': captions,
                'embeddings': embeddings, 'metadata': metadata}, output_path)
    meta_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f'Saved: {output_path}')
    print(f'Embedding shape: {tuple(embeddings.shape)}')
    print(f'Metadata: {meta_path}')


if __name__ == '__main__':
    main()

