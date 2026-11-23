#!/usr/bin/env python3
"""Run IDEALText validation-only lambda sweep on ND0 only.

Requires the accompanying small patch to run_ideal_text.py, which adds
--validation_only and writes validation_summary.json. Runs sequentially.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

DATASETS = ('ND0',)
WEIGHTS = tuple(round(i / 10, 1) for i in range(1, 10))


def save_tables(outdir, records):
    outdir.mkdir(parents=True, exist_ok=True)
    columns = ['dataset', 'lambda', 'best_validation_accuracy', 'best_epoch', 'epochs', 'status', 'checkpoint_dir', 'error']
    with (outdir / 'validation_runs.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)
    lookup = {(r['dataset'], float(r['lambda'])): r for r in records if r['status'] == 'ok'}
    with (outdir / 'validation_comparison.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['lambda', 'ND0_validation_accuracy', 'best_epoch'])
        for weight in WEIGHTS:
            row = lookup.get(('ND0', weight))
            writer.writerow([f'{weight:.1f}',
                             f"{float(row['best_validation_accuracy']):.4f}" if row else '',
                             row['best_epoch'] if row else ''])
    complete = [(w, float(lookup[('ND0', w)]['best_validation_accuracy']))
                for w in WEIGHTS if ('ND0', w) in lookup]
    if len(complete) == len(WEIGHTS):
        winner = max(complete, key=lambda x: x[1])
        with (outdir / 'recommended_lambda.txt').open('w') as f:
            f.write(f'Highest ND0 meta-validation accuracy: lambda={winner[0]:.1f}, '
                    f'accuracy={winner[1]:.4f}%\n'
                    'ND0 alone cannot establish stability across ND1-ND3.\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=400)
    p.add_argument('--bert_dir', default='./bert_features')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--output_dir', default='./lambda_validation_results')
    p.add_argument('--weights', nargs='+', type=float, default=list(WEIGHTS))
    args = p.parse_args()
    if args.epochs < 1:
        p.error('--epochs must be positive')
    root = Path(__file__).resolve().parent
    runner = root / 'run_ideal_text.py'
    if not runner.is_file():
        p.error(f'Place this script next to run_ideal_text.py: {runner}')
    outdir = (root / args.output_dir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    records = []
    for dataset in DATASETS:
        for weight in args.weights:
            weight = round(weight, 1)
            if weight not in WEIGHTS:
                p.error('Weights must be from 0.1 to 0.9 in 0.1 steps')
            ckpt_name = (f'Conv4_baseline++_conditionalsimsiam_IDEALText_IT_0_'
                         f'bilstm_0.1_0.1_lambda{weight}')
            ckpt = root / 'save' / 'checkpoints' / dataset / ckpt_name
            summary_path = ckpt / 'validation_summary.json'
            logpath = outdir / f'{dataset}_lambda{weight:.1f}.log'
            row = dict(dataset=dataset, **{'lambda': f'{weight:.1f}'},
                       best_validation_accuracy='', best_epoch='', epochs=args.epochs,
                       status='error', checkpoint_dir=str(ckpt), error='')
            print(f'\n===== {dataset}, lambda={weight:.1f}, epochs={args.epochs} =====', flush=True)
            if summary_path.is_file():
                try:
                    existing = json.loads(summary_path.read_text())
                    if int(existing.get('epochs', -1)) == args.epochs:
                        row.update(best_validation_accuracy=existing['best_validation_accuracy'],
                                   best_epoch=existing['best_epoch'], status='ok')
                        print('Using existing validation summary.', flush=True)
                        records.append(row)
                        save_tables(outdir, records)
                        continue
                except (ValueError, KeyError, TypeError):
                    pass
            if (ckpt / f'{args.epochs - 1}.tar').is_file():
                row['error'] = ('Completed checkpoint exists without matching validation summary. '
                                'Recover the validation value from the original training log, '
                                'or back up the checkpoint before rerunning.')
                records.append(row)
                save_tables(outdir, records)
                print(row['error'], flush=True)
                return 1
            cmd = [sys.executable, str(runner), '--dataset', dataset,
                   '--model_name', 'Conv4', '--noise_type', 'IT', '--noises', '0',
                   '--stop_epoch', str(args.epochs), '--attention_method', 'bilstm',
                   '--text_weight', str(weight), '--bert_dir', args.bert_dir,
                   '--device', args.device, '--validation_only']
            start = time.time()
            with logpath.open('w') as log:
                proc = subprocess.Popen(cmd, cwd=root, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, text=True, bufsize=1)
                for line in proc.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end='', flush=True)
                ret = proc.wait()
            if ret == 0 and summary_path.is_file():
                try:
                    result = json.loads(summary_path.read_text())
                    if int(result['epochs']) != args.epochs:
                        raise ValueError('Summary epoch count mismatch')
                    row.update(best_validation_accuracy=result['best_validation_accuracy'],
                               best_epoch=result['best_epoch'], status='ok')
                except (ValueError, KeyError, TypeError) as exc:
                    row['error'] = str(exc)
            else:
                row['error'] = f'Exit code {ret}; check {logpath}'
            records.append(row)
            save_tables(outdir, records)
            print(f'Finished {dataset} lambda={weight:.1f} in {(time.time()-start)/60:.1f} min: {row["status"]}', flush=True)
            if row['status'] != 'ok':
                print('Stopping on error to avoid wasting GPU time.', flush=True)
                return 1
    print(f'\nComplete. See {outdir / "validation_comparison.csv"}', flush=True)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
