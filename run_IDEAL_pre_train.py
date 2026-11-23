"""Baseline++ pre-training and ConditionalSimSiam SSL for ND0--ND3."""
import argparse
import os
import numpy as np
import torch

from utils.utils import *
from methods.conditionasimsiam import ConditionalSimSiam

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='ND0', choices=['ND0', 'ND1', 'ND2', 'ND3'])
parser.add_argument('--model_name', type=str, default='Conv4', choices=['Conv4', 'ResNet12'])
parser.add_argument('--train_n_way', type=int, default=2)
parser.add_argument('--test_n_way', type=int, default=2)
parser.add_argument('--n_shot', type=int, default=5)
parser.add_argument('--stop_epoch', type=int, default=-1,
                    help='Override both stages; -1 uses original IDEAL epoch schedules.')
parser.add_argument('--pre_stop_epoch', type=int, default=None,
                    help='Optional separate Baseline++ epoch count.')
parser.add_argument('--ssl_stop_epoch', type=int, default=None,
                    help='Optional separate ConditionalSimSiam epoch count.')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--pre_algorithm', type=str, default='baseline++')
parser.add_argument('--ssl_algorithm', type=str, default='conditionalsimsiam')
parser.add_argument('--stage', type=str, default='both', choices=['pre', 'ssl', 'both'])


def make_pre_model():
    return get_model(algorithm=pre_algorithm, model_name=model_name,
                     dataset=dataset, n_way=train_n_way,
                     n_shot=n_shot, adaptation=adaptation).to(device)


def make_ssl_model():
    return ConditionalSimSiam(model_func=model_dict[model_name],
                              n_way=train_n_way, n_support=n_shot,
                              image_size=image_size, device=device).to(device)


def load_checkpoint(model, directory, selector='resume'):
    path = get_resume_file(directory) if selector == 'resume' else get_best_file(directory)
    if path is None:
        raise FileNotFoundError(f'No {selector} checkpoint found in {directory}')
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['state'])
    print(f'Loaded checkpoint: {path}')
    return model


def pre_train():
    model = make_pre_model()
    optimizer = get_optimizer(model, model_name)
    for epoch in range(start_epoch, pre_epoch):
        model.train()
        model.train_loop(epoch, pre_base_loader, optimizer)
        if epoch == pre_epoch - 1:
            outfile = os.path.join(pre_dir, f'{epoch}.tar')
            torch.save({'epoch': epoch, 'state': model.state_dict(),
                        'dataset': dataset, 'backbone': model_name,
                        'stage': 'baseline++'}, outfile)
            print('Saved Baseline++ checkpoint:', outfile)


def ssl_train():
    sl_model = load_checkpoint(make_pre_model(), pre_dir, selector='resume')
    sl_model.eval()
    model = make_ssl_model()
    optimizer = get_optimizer(model, model_name)
    max_acc = float('-inf')

    for epoch in range(start_epoch, ssl_epoch):
        model.train()
        model.train_loop(epoch, ssl_base_loader, optimizer, sl_model=sl_model)

        model.eval()
        acc = model.test_loop(ssl_val_loader)
        if acc > max_acc:
            max_acc = acc
            outfile = os.path.join(ssl_dir, 'best_model.tar')
            torch.save({'epoch': epoch, 'state': model.state_dict(),
                        'dataset': dataset, 'backbone': model_name,
                        'stage': 'conditionalsimsiam', 'val_accuracy': float(acc)}, outfile)
            print(f'Best SSL validation accuracy: {acc:.2f}; saved {outfile}')

        if epoch == ssl_epoch - 1:
            outfile = os.path.join(ssl_dir, f'{epoch}.tar')
            torch.save({'epoch': epoch, 'state': model.state_dict(),
                        'dataset': dataset, 'backbone': model_name,
                        'stage': 'conditionalsimsiam'}, outfile)
            print('Saved final SSL checkpoint:', outfile)


if __name__ == '__main__':
    args = parser.parse_args()
    dataset = args.dataset
    model_name = args.model_name
    train_n_way = args.train_n_way
    test_n_way = args.test_n_way
    n_shot = args.n_shot
    device = args.device
    pre_algorithm = args.pre_algorithm
    ssl_algorithm = args.ssl_algorithm

    if train_n_way != 2 or test_n_way != 2 or n_shot != 5:
        raise ValueError('This experiment uses 2-way, 5-shot episodes.')
    if args.stop_epoch == 0 or args.pre_stop_epoch == 0 or args.ssl_stop_epoch == 0:
        raise ValueError('Epoch counts must be positive.')
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError(f'CUDA requested ({device}) but unavailable.')

    image_size = get_image_size(model_name=model_name, dataset=dataset)
    model_name = get_model_name(model_name=model_name, dataset=dataset)

    pre_epoch = (get_stop_epoch(algorithm=pre_algorithm, dataset=dataset, n_shot=n_shot)
                 if args.stop_epoch == -1 else args.stop_epoch)
    ssl_epoch = (get_stop_epoch(algorithm=ssl_algorithm, dataset=dataset, n_shot=n_shot)
                 if args.stop_epoch == -1 else args.stop_epoch)
    if args.pre_stop_epoch is not None:
        pre_epoch = args.pre_stop_epoch
    if args.ssl_stop_epoch is not None:
        ssl_epoch = args.ssl_stop_epoch
    if pre_epoch < 1 or ssl_epoch < 1:
        raise ValueError('Epoch counts must be positive.')

    pre_dir = os.path.join(base_path, 'save', 'checkpoints', dataset,
                           f'{model_name}_{pre_algorithm}')
    ssl_dir = os.path.join(base_path, 'save', 'checkpoints', dataset,
                           f'{model_name}_{pre_algorithm}_{ssl_algorithm}')
    os.makedirs(pre_dir, exist_ok=True)
    os.makedirs(ssl_dir, exist_ok=True)

    base_file, val_file = get_train_files(dataset=dataset)
    for path in (base_file, val_file):
        if not os.path.isfile(path):
            raise FileNotFoundError(f'Missing dataset JSON: {path}')

    print(f'Dataset={dataset}, Backbone={model_name}, Way={train_n_way}, Shot={n_shot}')
    print(f'Baseline++ epochs={pre_epoch}, SSL epochs={ssl_epoch}, Device={device}')
    print(f'Base={base_file}\nValidation={val_file}')

    if args.stage in ('pre', 'both'):
        pre_base_loader, _ = get_train_loader(
            algorithm=pre_algorithm, image_size=image_size,
            base_file=base_file, val_file=val_file,
            train_n_way=train_n_way, test_n_way=test_n_way,
            n_shot=n_shot, num_workers=num_workers)
        final_pre = os.path.join(pre_dir, f'{pre_epoch - 1}.tar')
        if os.path.isfile(final_pre):
            print('Baseline++ final checkpoint already exists:', final_pre)
        else:
            print('Starting Baseline++ pre-training')
            pre_train()

    if args.stage in ('ssl', 'both'):
        if get_resume_file(pre_dir) is None:
            raise FileNotFoundError('Baseline++ checkpoint is required before SSL training: ' + pre_dir)
        ssl_base_loader, ssl_val_loader = get_train_loader(
            algorithm=ssl_algorithm, image_size=image_size,
            base_file=base_file, val_file=val_file,
            train_n_way=train_n_way, test_n_way=test_n_way,
            n_shot=n_shot, num_workers=num_workers)
        final_ssl = os.path.join(ssl_dir, f'{ssl_epoch - 1}.tar')
        if os.path.isfile(final_ssl) and get_best_file(ssl_dir) is not None:
            print('SSL checkpoints already exist:', ssl_dir)
        else:
            print('Starting ConditionalSimSiam training')
            ssl_train()

    print('Selected stage(s) completed.')

