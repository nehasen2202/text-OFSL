#!/usr/bin/env python3
"""2x2 confusion matrices for ND0–ND3, using the current IDEALText pipeline.
Run from the IDEAL project root (same directory as run_ideal_text.py).
The script DOES inject noise exactly as IDEALText.test_loop does.
"""
import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix

from utils.utils import (base_path, model_dict, get_image_size, get_model_name,
                         get_train_files, get_novel_file, get_best_file)
from methods.conditionasimsiam import ConditionalSimSiam
from methods.IDEAL_Text import IDEALText, BertFeatureStore
from data.datamgr_text import SetDataManagerText

# The two *actual* novel classes, in requested display order.
# ND2/ND3 codes are resolved against class names or parent folder names in novel.json.
EXPECTED = {
    'ND0': [('snowstorm_blizzard', 'Snowstorm Blizzard'), ('fire', 'Fire')],
    'ND1': [('flood', 'Flood'), ('hurricane_typhoon', 'Hurricane Typhoon')],
    'ND2': [('n01211111', 'Earthquake'), ('n01211112', 'Flood')],
    'ND3': [('C0000', 'Earthquake'), ('C0001', 'Landslide')],
}


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_json(path):
    with open(path, encoding='utf-8') as handle:
        return json.load(handle)


def resolve_class_labels(dataset, meta):
    """Match semantic/code identifiers to real global labels, never guess numeric IDs."""
    labels = meta['image_labels']
    paths = meta['image_names']
    if len(labels) != len(paths):
        raise ValueError('novel.json image_labels/image_names length mismatch')
    unique = list(dict.fromkeys(labels))
    if len(unique) != 2:
        raise ValueError(f'{dataset}: expected exactly 2 novel classes, got {unique}')
    names = meta.get('class_names') or meta.get('label_names') or meta.get('classes')
    aliases = {str(label): {str(label).casefold()} for label in unique}
    for label, path in zip(labels, paths):
        aliases[str(label)].add(Path(path).parent.name.casefold())
    if isinstance(names, dict):
        for label, name in names.items():
            if str(label) in aliases:
                aliases[str(label)].add(str(name).casefold())
    elif isinstance(names, list):
        if len(names) == len(unique):
            for label, name in zip(unique, names):
                aliases[str(label)].add(str(name).casefold())
        elif len(names) == len(labels):
            for label, name in zip(labels, names):
                aliases[str(label)].add(str(name).casefold())
        elif all(isinstance(v, int) and 0 <= v < len(names) for v in unique):
            for label in unique:
                aliases[str(label)].add(str(names[label]).casefold())
    result = []
    for identifier, display in EXPECTED[dataset]:
        hits = [label for label in unique
                if identifier.casefold() in aliases[str(label)]]
        if len(hits) != 1:
            raise ValueError(
                f'{dataset}: cannot uniquely resolve {identifier!r} to a novel.json label. '
                f'Available labels/aliases: {aliases}. '
                'Check folder names, class_names or label_names in novel.json; '
                'do not silently substitute numeric labels.')
        result.append((hits[0], display))
    if result[0][0] == result[1][0]:
        raise ValueError(f'{dataset}: both requested classes resolved to the same label')
    return result


def checkpoint_dir(dataset, internal_name, args):
    # OOD evaluation uses OOT-trained checkpoints, exactly as run_ideal_text.py.
    ck_noise = 'OOT' if args.noise_type == 'OOD' else args.noise_type
    return (Path(base_path) / 'save' / 'checkpoints' / dataset /
            f'{internal_name}_{args.pre_algorithm}_{args.ssl_algorithm}_'
            f'IDEALText_{ck_noise}_{args.noises}_'
            f'{args.attention_method}_{args.eta}_{args.gamma}_lambda{args.text_weight}')


def make_model(dataset, args, store, train_meta, noise_meta):
    image_size = get_image_size(model_name=args.model_name, dataset=dataset)
    internal_name = get_model_name(model_name=args.model_name, dataset=dataset)
    ssl_dir = (Path(base_path) / 'save' / 'checkpoints' / dataset /
               f'{internal_name}_{args.pre_algorithm}_{args.ssl_algorithm}')
    ssl_ckpt = get_best_file(str(ssl_dir))
    if ssl_ckpt is None:
        raise FileNotFoundError(f'SSL checkpoint missing in {ssl_dir}')
    ssl = ConditionalSimSiam(model_func=model_dict[internal_name], n_way=2,
                             n_support=5, image_size=image_size, device=args.device)
    ssl.load_state_dict(torch.load(ssl_ckpt, map_location=args.device, weights_only=False)['state'])
    ssl.eval()
    model = IDEALText(
        model_func=model_dict[internal_name], n_way=2, n_support=5,
        image_size=image_size, outlier=args.noises,
        image_files=np.asarray(train_meta['image_names']),
        image_labels=np.asarray(train_meta['image_labels']),
        noise_type=args.noise_type, ssl_feature_extractor=ssl.feature_extractor,
        attention_method=args.attention_method, eta=args.eta, gamma=args.gamma,
        device=args.device, feature_store=store, text_weight=args.text_weight)
    ckpt_dir = checkpoint_dir(dataset, internal_name, args)
    model_ckpt = get_best_file(str(ckpt_dir))
    if model_ckpt is None:
        raise FileNotFoundError(f'IDEALText checkpoint missing in {ckpt_dir}')
    model.load_state_dict(torch.load(model_ckpt, map_location=args.device,
                                     weights_only=False)['state'])
    model.image_files = np.asarray(noise_meta['image_names'])
    model.image_labels = np.asarray(noise_meta['image_labels'])
    model.eval()
    print(f'  SSL: {ssl_ckpt}\n  IDEALText: {model_ckpt}')
    return model, image_size


def evaluate(model, loader, label_pairs):
    global_to_index = {str(label): i for i, (label, _) in enumerate(label_pairs)}
    rows, episode_accuracy = [], []
    with torch.no_grad():
        for ep, (x, y, paths) in enumerate(loader, 1):
            model.n_way = x.size(0)
            model.n_query = x.size(1) - model.n_support
            if model.n_way != 2 or model.n_query != 15:
                raise ValueError('Expected 2-way, 5-shot, 15 queries/class')
            if model.image_size is None:
                model.image_size = x.shape[-1]
            # This step is essential: inject IT/OOT/OOD support noise and update
            # image paths so the BERT text vectors align with replacement images.
            x, noisy_paths = model.add_noise_with_paths(
                x, y, paths, model.noise_type, aug=False)
            model._active_paths = noisy_paths
            scores = model.set_forward(x.to(model.device))
            predictions = scores.argmax(dim=1).cpu().numpy()
            episode_labels = [str(y[i, 0].item()) for i in range(2)]
            if set(episode_labels) != set(global_to_index):
                raise ValueError(f'Unexpected episode labels: {episode_labels}')
            correct = []
            for row in range(2):
                for query in range(model.n_query):
                    position = row * model.n_query + query
                    true_idx = global_to_index[episode_labels[row]]
                    pred_idx = global_to_index[episode_labels[int(predictions[position])]]
                    correct.append(true_idx == pred_idx)
                    rows.append({'episode': ep, 'true_class': label_pairs[true_idx][1],
                                 'predicted_class': label_pairs[pred_idx][1],
                                 'true_global_label': episode_labels[row],
                                 'predicted_global_label': episode_labels[int(predictions[position])],
                                 'correct': true_idx == pred_idx})
            episode_accuracy.append(100 * np.mean(correct))
    model._active_paths = None
    return pd.DataFrame(rows), np.asarray(episode_accuracy)


def draw(ax, percentages, names, title):
    im = ax.imshow(percentages, vmin=0, vmax=100, cmap='Blues', aspect='equal')
    ax.set_xticks([0, 1], labels=names)
    ax.set_yticks([0, 1], labels=names)
    ax.set_xlabel('Predicted class')
    ax.set_ylabel('True class')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', labelrotation=20)
    for i in range(2):
        for j in range(2):
            v = percentages[i, j]
            ax.text(j, i, f'{v:.1f}%', ha='center', va='center',
                    fontsize=13, fontweight='bold',
                    color='white' if v >= 50 else 'black')
    ax.set_xticks(np.arange(-.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 2, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=1.5)
    ax.tick_params(which='minor', bottom=False, left=False)
    return im


def run_dataset(dataset, args):
    print(f'\n=== {dataset}: {args.noise_type}, {args.noises}/5 support replacements ===')
    seed_all(args.seed)
    base_file, val_file = get_train_files(dataset=dataset)
    novel_file = get_novel_file(dataset=dataset, split='novel')
    train_meta, novel_meta = load_json(base_file), load_json(novel_file)
    label_pairs = resolve_class_labels(dataset, novel_meta)
    print('  Novel classes:', label_pairs)
    if args.noise_type == 'OOD':
        ood_json = Path(base_path) / 'filelists' / 'miniImagenet' / 'novel.json'
        noise_meta = load_json(ood_json)
    else:
        noise_meta = train_meta
    nd_bert = Path(args.bert_dir) / f'{dataset}_BERT_embeddings.pt'
    bert_files = [str(nd_bert)]
    if args.noise_type == 'OOD' and args.noises > 0:
        if not args.ood_bert:
            raise ValueError('OOD requires --ood_bert path for miniImageNet captions')
        bert_files.append(args.ood_bert)
    store = BertFeatureStore(bert_files)
    store.verify_json(novel_file)
    if args.noise_type == 'OOT' and args.noises:
        store.verify_json(base_file)
    if args.noise_type == 'OOD' and args.noises:
        store.verify_json(str(ood_json))
    model, image_size = make_model(dataset, args, store, train_meta, noise_meta)
    loader = SetDataManagerText(image_size, n_way=2, n_support=5, n_query=15,
                                n_eposide=args.num_episodes,
                                num_workers=args.num_workers).get_data_loader(novel_file, aug=False)
    pred, episode_acc = evaluate(model, loader, label_pairs)
    names = [name for _, name in label_pairs]
    counts = confusion_matrix(pred.true_class, pred.predicted_class, labels=names)
    if counts.shape != (2, 2) or np.any(counts.sum(axis=1) == 0):
        raise ValueError('Expected nonempty 2x2 confusion matrix')
    pct = counts / counts.sum(axis=1, keepdims=True) * 100
    accuracy = float(np.mean(episode_acc))
    ci95 = float(1.96 * np.std(episode_acc) / np.sqrt(len(episode_acc)))
    out = Path(args.output_dir) / dataset
    out.mkdir(parents=True, exist_ok=True)
    prefix = (f'{dataset}_{args.model_name}_{args.attention_method}_'
              f'{args.noise_type}_noise{args.noises}_2way_5shot')
    pred.to_csv(out / f'{prefix}_all_predictions.csv', index=False)
    pd.DataFrame(counts, index=names, columns=names).to_csv(out / f'{prefix}_confusion_counts.csv')
    pd.DataFrame(pct, index=names, columns=names).to_csv(out / f'{prefix}_confusion_percentages.csv')
    pd.DataFrame({'class': names, 'correct': np.diag(counts),
                  'total': counts.sum(axis=1), 'accuracy_percent': np.diag(pct)}).to_csv(
                      out / f'{prefix}_per_class_accuracy.csv', index=False)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = draw(ax, pct, names, f'{dataset}: IDEAL+Text ({args.noise_type}, {args.noises * 20}% noise)')
    fig.colorbar(im, ax=ax, label='Percentage of true-class queries (%)')
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(out / f'{prefix}_confusion_matrix.{ext}', dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f'  Accuracy: {accuracy:.2f} ± {ci95:.2f}% | predictions: {len(pred)}')
    return {'dataset': dataset, 'pct': pct, 'names': names, 'accuracy': accuracy, 'ci95': ci95}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=['all', *EXPECTED], default='all')
    p.add_argument('--noise_type', choices=['IT', 'OOT', 'OOD'], default='OOT')
    p.add_argument('--noises', type=int, choices=[0, 1, 2], default=1)
    p.add_argument('--model_name', choices=['Conv4', 'ResNet12'], default='Conv4')
    p.add_argument('--attention_method', choices=['bilstm', 'transformer'], default='bilstm')
    p.add_argument('--pre_algorithm', default='baseline++')
    p.add_argument('--ssl_algorithm', default='conditionalsimsiam')
    p.add_argument('--eta', type=float, default=0.1)
    p.add_argument('--gamma', type=float, default=0.1)
    p.add_argument('--text_weight', type=float, default=0.7)
    p.add_argument('--bert_dir', default='./bert_features')
    p.add_argument('--ood_bert', default=None)
    p.add_argument('--num_episodes', type=int, default=600)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output_dir', default='confusion_matrix_outputs')
    args = p.parse_args()
    if args.noise_type == 'OOD' and args.noises == 0:
        p.error('For clean evaluation, use --noise_type IT --noises 0')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        args.device = 'cpu'
    datasets = list(EXPECTED) if args.dataset == 'all' else [args.dataset]
    results = [run_dataset(ds, args) for ds in datasets]
    if len(results) > 1:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        for ax, result in zip(axes.flat, results):
            draw(ax, result['pct'], result['names'], result['dataset'])
        fig.suptitle(f'IDEAL+Text: {args.noise_type} ({args.noises * 20}% noise)', fontsize=16)
        fig.tight_layout()
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        for ext in ('png', 'pdf'):
            fig.savefig(output / f'combined_confusion_matrices.{ext}', dpi=600, bbox_inches='tight')
        plt.close(fig)
    pd.DataFrame([{'dataset': r['dataset'], 'accuracy_percent': r['accuracy'],
                   'ci95': r['ci95']} for r in results]).to_csv(
                       Path(args.output_dir) / 'confusion_matrix_summary.csv', index=False)


if __name__ == '__main__':
    main()

