import os
import json
import argparse

from utils.utils import *
from methods.conditionasimsiam import ConditionalSimSiam
from methods.IDEAL_Text import IDEALText, BertFeatureStore
from data.datamgr_text import SetDataManagerText, get_train_loader_text


parser = argparse.ArgumentParser()

parser.add_argument(
    '--dataset',
    type=str,
    default='ND0',
    choices=['ND0', 'ND1', 'ND2', 'ND3']
)

parser.add_argument(
    '--noises',
    type=int,
    default=1,
    choices=[0, 1, 2],
    help='Number of corrupted support images per class'
)

parser.add_argument(
    '--noise_type',
    type=str,
    default='OOT',
    choices=['IT', 'OOT', 'OOD']
)

parser.add_argument(
    '--model_name',
    type=str,
    default='Conv4',
    choices=['Conv4', 'ResNet12']
)

parser.add_argument('--train_n_way', type=int, default=2)
parser.add_argument('--test_n_way', type=int, default=2)
parser.add_argument('--n_shot', type=int, default=5)

parser.add_argument('--stop_epoch', type=int, default=-1)
parser.add_argument('--device', type=str, default='cuda:0')

parser.add_argument('--pre_algorithm', type=str, default='baseline++')
parser.add_argument('--ssl_algorithm', type=str, default='conditionalsimsiam')
parser.add_argument('--meta_algorithm', type=str, default='IDEAL')

parser.add_argument(
    '--attention_method',
    type=str,
    default='bilstm',
    choices=['bilstm', 'transformer']
)

parser.add_argument('--eta', type=float, default=0.1)
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--text_weight', type=float, default=0.3)
parser.add_argument('--validation_only', action='store_true', help='Save best validation results and skip meta-test')
parser.add_argument('--bert_dir', type=str, default='./bert_features')
parser.add_argument('--ood_bert', type=str, default=None, help='Path to miniImageNet BERT .pt (required for OOD)')


def meta_train():

    ssl_model = ConditionalSimSiam(
        model_func=model_dict[model_name],
        n_way=train_n_way,
        n_support=n_shot,
        image_size=image_size,
        device=device
    )

    modelfile = get_best_file(ssl_dir)

    assert modelfile is not None, (
        f"SSL checkpoint not found: {ssl_dir}"
    )

    tmp = torch.load(modelfile, map_location=device)
    ssl_model.load_state_dict(tmp['state'])
    ssl_model.eval()

    model = IDEALText(
        model_func=model_dict[model_name],
        n_way=train_n_way,
        n_support=n_shot,
        image_size=image_size,
        outlier=outlier,
        image_files=train_files,
        image_labels=train_labels,
        noise_type=noise_type,
        ssl_feature_extractor=ssl_model.feature_extractor,
        attention_method=attention_method,
        eta=eta,
        gamma=gamma,
        device=device,
        feature_store=feature_store,
        text_weight=text_weight
    )

    optimizer = get_optimizer(model, model_name)

    max_acc = -1.0
    best_epoch = -1

    for epoch in range(start_epoch, meta_epoch):

        model.train()

        # Training OOT pool: base.json
        # IT noise uses the current episode internally.
        model.train_loop(
            epoch,
            meta_base_loader,
            optimizer,
            image_files=train_files,
            image_labels=train_labels
        )

        model.eval()

        # Validation OOT pool: base.json
        # Validation episodes themselves come from val.json.
        acc = model.test_loop(
            meta_val_loader,
            image_files=train_files,
            image_labels=train_labels
        )

        if acc > max_acc:

            print(acc, "--> Best model! save...")

            max_acc = acc
            best_epoch = epoch

            outfile = os.path.join(
                meta_dir,
                'best_model.tar'
            )

            torch.save(
                {
                    'epoch': epoch,
                    'state': model.state_dict()
                },
                outfile
            )

        if epoch == meta_epoch - 1:

            outfile = os.path.join(
                meta_dir,
                '{:d}.tar'.format(epoch)
            )

            torch.save(
                {
                    'epoch': epoch,
                    'state': model.state_dict()
                },
                outfile
            )

    summary = {
        'dataset': dataset, 'text_weight': text_weight,
        'attention_method': attention_method, 'model': model_name,
        'noise_type': noise_type, 'noises': outlier,
        'best_validation_accuracy': float(max_acc),
        'best_epoch': int(best_epoch), 'epochs': int(meta_epoch)
    }
    summary_path = os.path.join(meta_dir, 'validation_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print('Best validation accuracy:', max_acc, '| epoch:', best_epoch)


def meta_test():

    ssl_model = ConditionalSimSiam(
        model_func=model_dict[model_name],
        n_way=train_n_way,
        n_support=n_shot,
        image_size=image_size,
        device=device
    )

    modelfile = get_best_file(ssl_dir)

    assert modelfile is not None, (
        f"SSL checkpoint not found: {ssl_dir}"
    )

    tmp = torch.load(modelfile, map_location=device)

    ssl_model.load_state_dict(tmp['state'])
    ssl_model.eval()

    model = IDEALText(
        model_func=model_dict[model_name],
        n_way=test_n_way,
        n_support=n_shot,
        image_size=image_size,
        outlier=outlier,
        image_files=train_files,
        image_labels=train_labels,
        noise_type=noise_type,
        ssl_feature_extractor=ssl_model.feature_extractor,
        attention_method=attention_method,
        eta=eta,
        gamma=gamma,
        device=device,
        feature_store=feature_store,
        text_weight=text_weight
    )

    modelfile = get_best_file(checkpoint_dir)

    assert modelfile is not None, (
        f"IDEAL checkpoint not found: {checkpoint_dir}"
    )

    tmp = torch.load(modelfile, map_location=device)

    model.load_state_dict(tmp['state'])
    model.eval()

    # Test episodes always come from NDx/novel.json.
    # OOT uses NDx/base.json; OOD uses miniImageNet images.
    results = model.test_loop(
        novel_loader,
        return_metrics=True,
        image_files=test_noise_files,
        image_labels=test_noise_labels
    )
    print("Final evaluation:", results)
    output_path = os.path.join(meta_dir, 'test_metrics.json')
    with open(output_path, 'w') as f:
        json.dump({
            'dataset': dataset,
            'noise_type': noise_type,
            'noises': outlier,
            'model': model_name,
            'attention': attention_method,
            'text_weight': text_weight,
            'text_model': 'frozen bert-base-uncased; mean-pooled last hidden state',
            'metrics': results
        }, f, indent=2)
    print("Saved metrics:", output_path)


if __name__ == '__main__':

    args = parser.parse_args()

    dataset = args.dataset
    outlier = args.noises
    noise_type = args.noise_type

    model_name = args.model_name

    train_n_way = args.train_n_way
    test_n_way = args.test_n_way
    n_shot = args.n_shot

    stop_epoch = args.stop_epoch
    device = args.device

    pre_algorithm = args.pre_algorithm
    ssl_algorithm = args.ssl_algorithm
    meta_algorithm = args.meta_algorithm

    attention_method = args.attention_method

    eta = args.eta
    gamma = args.gamma
    text_weight = args.text_weight

    # --------------------------------------------------
    # Validate experimental configuration
    # --------------------------------------------------

    if train_n_way != 2 or test_n_way != 2:
        raise ValueError(
            "ND0-ND3 experiments require 2-way episodes."
        )

    if n_shot != 5:
        raise ValueError(
            "Current experiments require 5-shot episodes."
        )

    if noise_type == 'OOD' and outlier == 0:
        raise ValueError('Use IT or OOT with --noises 0 for clean evaluation.')

    # --------------------------------------------------
    # Model configuration
    # --------------------------------------------------

    image_size = get_image_size(
        model_name=model_name,
        dataset=dataset
    )

    model_name = get_model_name(
        model_name=model_name,
        dataset=dataset
    )

    if stop_epoch == -1:

        meta_epoch = get_stop_epoch(
            algorithm=meta_algorithm,
            dataset=dataset,
            n_shot=n_shot
        )

    else:
        meta_epoch = stop_epoch

    # --------------------------------------------------
    # SSL checkpoint directory
    # --------------------------------------------------

    ssl_dir = (
        base_path
        + f'/save/checkpoints/{dataset}/'
        + f'{model_name}_{pre_algorithm}_{ssl_algorithm}'
    )

    # --------------------------------------------------
    # Load training and validation JSON files
    # --------------------------------------------------

    base_file, val_file = get_train_files(
        dataset=dataset
    )

    with open(base_file, 'r') as f:
        base_data = json.load(f)

    train_labels = np.array(base_data['image_labels'])
    train_files = np.array(base_data['image_names'])

    with open(val_file, 'r') as f:
        val_data = json.load(f)

    val_labels = np.array(val_data['image_labels'])
    val_files = np.array(val_data['image_names'])

    # --------------------------------------------------
    # Create training and validation episode loaders
    # --------------------------------------------------

    meta_base_loader, meta_val_loader = get_train_loader_text(
        image_size=image_size,
        base_file=base_file,
        val_file=val_file,
        train_n_way=train_n_way,
        test_n_way=test_n_way,
        n_shot=n_shot,
        num_workers=num_workers
    )

    # --------------------------------------------------
    # Create novel/test episode loader
    # --------------------------------------------------

    novel_file = get_novel_file(
        dataset=dataset,
        split='novel'
    )

    datamgr = SetDataManagerText(
        image_size,
        n_eposide=test_iter_num,
        n_query=15,
        n_way=test_n_way,
        n_support=n_shot,
        num_workers=num_workers
    )

    novel_loader = datamgr.get_data_loader(
        novel_file,
        aug=False
    )

    # OOD: use a fixed miniImageNet split for external contamination.
    # The corresponding OOT-trained model is loaded for evaluation.
    test_noise_files = train_files
    test_noise_labels = train_labels
    if noise_type == 'OOD':
        ood_json = os.path.join(base_path, 'filelists', 'miniImagenet', 'novel.json')
        if not os.path.isfile(ood_json):
            raise FileNotFoundError(
                f"miniImageNet OOD JSON not found: {ood_json}. "
                "Check the folder name and image paths."
            )
        with open(ood_json, 'r') as f:
            ood_data = json.load(f)
        test_noise_files = np.asarray(ood_data['image_names'])
        test_noise_labels = np.asarray(ood_data['image_labels'])
        if len(test_noise_files) == 0:
            raise ValueError('miniImageNet OOD pool is empty')

    # Load frozen text features; validate all paths before starting training.
    bert_dir = os.path.abspath(args.bert_dir)
    nd_bert = os.path.join(bert_dir, f'{dataset}_BERT_embeddings.pt')
    if not os.path.isfile(nd_bert):
        raise FileNotFoundError(f'ND BERT file missing: {nd_bert}')
    bert_files = [nd_bert]
    if noise_type == 'OOD':
        if not args.ood_bert:
            raise ValueError('--ood_bert is required for OOD (miniImageNet caption embeddings)')
        bert_files.append(args.ood_bert)
    feature_store = BertFeatureStore(bert_files)
    for filename in (base_file, val_file, novel_file):
        feature_store.verify_json(filename)
    if noise_type == 'OOD' and outlier > 0:
        feature_store.verify_json(ood_json)

    print("\n========================================")
    print("IDEAL Experiment Configuration")
    print("========================================")
    print("Dataset:", dataset)
    print("Model:", model_name)
    print("Train way:", train_n_way)
    print("Test way:", test_n_way)
    print("Shot:", n_shot)
    print("Noise type:", noise_type)
    print("Noise samples per class:", outlier)
    print("Attention:", attention_method)
    print("Eta:", eta)
    print("Gamma:", gamma)
    print("BERT text weight:", text_weight)
    print("Training JSON:", base_file)
    print("Validation JSON:", val_file)
    print("Testing JSON:", novel_file)

    if noise_type == 'OOT':
        print("OOT source:", base_file)
    elif noise_type == 'OOD':
        print("OOD source:", ood_json)
    else:
        print("IT source: current episode (or clean when noises=0)")

    print("========================================\n")

    # --------------------------------------------------
    # IDEAL checkpoint directory
    # --------------------------------------------------

    meta_dir = (
        base_path
        + f'/save/checkpoints/{dataset}/'
        + f'{model_name}_{pre_algorithm}_{ssl_algorithm}_'
        + f'IDEALText_{noise_type}_{outlier}_'
        + f'{attention_method}_{eta}_{gamma}_lambda{text_weight}'
    )

    # OOD testing uses the matching OOT-trained checkpoint, not OOD training.
    checkpoint_dir = meta_dir
    if noise_type == 'OOD':
        checkpoint_dir = (
            base_path + f'/save/checkpoints/{dataset}/'
            + f'{model_name}_{pre_algorithm}_{ssl_algorithm}_'
            + f'IDEALText_OOT_{outlier}_'
            + f'{attention_method}_{eta}_{gamma}_lambda{text_weight}'
        )

    os.makedirs(meta_dir, exist_ok=True)

    # --------------------------------------------------
    # Meta-training
    # --------------------------------------------------

    print(
        dataset,
        meta_algorithm,
        attention_method,
        model_name,
        train_n_way,
        n_shot,
        noise_type,
        outlier,
        eta,
        gamma,
        'Start meta-training!'
    )

    final_checkpoint = os.path.join(
        meta_dir,
        '{:d}.tar'.format(meta_epoch - 1)
    )

    if noise_type == 'OOD':
        print('OOD is evaluation-only; loading OOT checkpoint from:', checkpoint_dir)
        if not os.path.isfile(os.path.join(checkpoint_dir, 'best_model.tar')):
            raise FileNotFoundError('Train the matching OOT experiment first: ' + checkpoint_dir)
    elif os.path.exists(final_checkpoint):
        if args.validation_only and not os.path.isfile(os.path.join(meta_dir, 'validation_summary.json')):
            raise RuntimeError('Existing final checkpoint has no validation summary. Back up checkpoint and rerun, or recover validation accuracy from logs.')
        print('Using existing model!')
    else:
        meta_train()

    # --------------------------------------------------
    # Meta-testing
    # --------------------------------------------------

    print(
        dataset,
        meta_algorithm,
        attention_method,
        model_name,
        test_n_way,
        n_shot,
        noise_type,
        outlier,
        eta,
        gamma,
        'Start meta-testing!'
    )

    if args.validation_only:
        print('Validation-only: skipping meta-testing on held-out novel classes.')
    else:
        meta_test()
