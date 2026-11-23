import os
import argparse
from utils.utils import *
from methods.cosine_baseline import CosineBaseline

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cifar',
                    choices=['ND0', 'ND1', 'ND2', 'ND3'])
parser.add_argument('--model_name', type=str, default='Conv4',
                    choices=['Conv4', 'ResNet12'])
parser.add_argument('--train_n_way', type=int, default=2)
parser.add_argument('--test_n_way', type=int, default=2)
parser.add_argument('--n_shot', type=int, default=5)
parser.add_argument('--stop_epoch', type=int, default=-1)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--meta_algorithm', type=str, default='Cosine')
parser.add_argument('--temperature', type=float, default=10.0)


def meta_train():
    model = CosineBaseline(
        model_func=model_dict[model_name],
        n_way=train_n_way,
        n_support=n_shot,
        temperature=temperature,
        device=device
    )
    model = model.to(device)
    optimizer = get_optimizer(model, model_name)

    max_acc = 0
    for epoch in range(start_epoch, meta_epoch):
        model.train()
        model.train_loop(epoch, meta_base_loader, optimizer)

        model.eval()
        acc = model.test_loop(meta_val_loader)

        if acc > max_acc:
            print(acc, "--> Best model! save...")
            max_acc = acc
            outfile = os.path.join(meta_dir, 'best_model.tar')
            torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)

        if epoch == meta_epoch - 1:
            outfile = os.path.join(meta_dir, '{:d}.tar'.format(epoch))
            torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)


def meta_test():
    model = CosineBaseline(
        model_func=model_dict[model_name],
        n_way=test_n_way,
        n_support=n_shot,
        temperature=temperature,
        device=device
    )
    modelfile = get_best_file(meta_dir)
    assert modelfile is not None
    tmp = torch.load(modelfile)
    model.load_state_dict(tmp['state'])
    model = model.to(device)
    model.eval()

    acc_mean, acc_std = model.test_loop(novel_loader, return_std=True)
    print('%d Test Acc = %4.2f±%4.2f%%' %
          (test_iter_num, acc_mean, 1.96 * acc_std / np.sqrt(test_iter_num)))


if __name__ == '__main__':
    args = parser.parse_args()
    dataset = args.dataset
    model_name = args.model_name
    train_n_way = args.train_n_way
    test_n_way = args.test_n_way
    n_shot = args.n_shot
    stop_epoch = args.stop_epoch
    device = args.device
    meta_algorithm = args.meta_algorithm
    temperature = args.temperature

    image_size = get_image_size(model_name=model_name, dataset=dataset)
    model_name = get_model_name(model_name=model_name, dataset=dataset)

    if stop_epoch == -1:
        meta_epoch = get_stop_epoch(algorithm=meta_algorithm, dataset=dataset, n_shot=n_shot)
    else:
        meta_epoch = stop_epoch

    base_file, val_file = get_train_files(dataset=dataset)
    meta_base_loader, meta_val_loader = get_train_loader(
        algorithm=meta_algorithm,
        image_size=image_size,
        base_file=base_file,
        val_file=val_file,
        train_n_way=train_n_way,
        test_n_way=test_n_way,
        n_shot=n_shot,
        num_workers=num_workers
    )

    datamgr = SetDataManager(
        image_size,
        n_eposide=test_iter_num,
        n_query=15,
        n_way=test_n_way,
        n_support=n_shot,
        num_workers=num_workers
    )
    novel_file = get_novel_file(dataset=dataset, split='novel')
    novel_loader = datamgr.get_data_loader(novel_file, aug=False)

    meta_dir = get_checkpoint_dir(
        algorithm=meta_algorithm,
        model_name=model_name,
        dataset=dataset,
        train_n_way=train_n_way,
        n_shot=n_shot
    )

    print(dataset, meta_algorithm, model_name, train_n_way, n_shot, 'Start meta-training!')
    if os.path.exists(os.path.join(meta_dir, '{:d}.tar'.format(meta_epoch - 1))):
        print('Using exist model!')
    else:
        meta_train()

    print(dataset, meta_algorithm, model_name, test_n_way, n_shot, 'Start meta-testing!')
    meta_test()
