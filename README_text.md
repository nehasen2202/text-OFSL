# IDEAL + frozen BERT caption features (separate experimental pipeline)

Copy `data/*.py`, `methods/*.py`, and `run_ideal_text.py` into the corresponding directories in your **existing** IDEAL project. Do not replace any original files.

## Prerequisites

- Your original IDEAL repository and its working `run_ideal.py` environment.
- Existing baseline++ and ConditionalSimSiam pretrained checkpoints (reused unchanged).
- Saved caption-derived BERT embeddings: `bert_features/ND0_BERT_embeddings.pt` etc. Each `.pt` must have `image_paths` and `embeddings` keys; embeddings shape `[N, 768]`.
- For OOD, one separate miniImageNet BERT `.pt` with paths exactly matching the images listed in `filelists/miniImagenet/novel.json`.
- Existing class-disjoint `base.json`, `val.json`, and `novel.json` for each ND dataset.

## Run from project root

Smoke test (1 epoch, zero-noise):

```bash
python run_ideal_text.py --dataset ND0 --model_name Conv4 --noise_type IT --noises 0 --stop_epoch 1 --text_weight 0.3 --bert_dir ./bert_features
```

**IMPORTANT:** The smoke test saves an epoch-0 checkpoint under its own directory. For a full 400-epoch run, use `--stop_epoch 400`; because the runner checks only the final epoch checkpoint, this starts a new training run in the same directory. To keep smoke-test artifacts distinct, move or delete the smoke-test checkpoint directory before full training. No existing image-only IDEAL directory is touched.

Full 400 epochs (2-way 5-shot, 100 episodes/epoch, 600 test episodes):

```bash
python run_ideal_text.py --dataset ND0 --model_name Conv4 --noise_type IT --noises 0 --stop_epoch 400 --text_weight 0.3 --bert_dir ./bert_features
python run_ideal_text.py --dataset ND0 --model_name Conv4 --noise_type IT --noises 1 --stop_epoch 400 --text_weight 0.3 --bert_dir ./bert_features
python run_ideal_text.py --dataset ND0 --model_name Conv4 --noise_type IT --noises 2 --stop_epoch 400 --text_weight 0.3 --bert_dir ./bert_features
python run_ideal_text.py --dataset ND0 --model_name Conv4 --noise_type OOT --noises 1 --stop_epoch 400 --text_weight 0.3 --bert_dir ./bert_features
python run_ideal_text.py --dataset ND0 --model_name Conv4 --noise_type OOT --noises 2 --stop_epoch 400 --text_weight 0.3 --bert_dir ./bert_features
```

OOD is **evaluation only** and loads the matching text-augmented OOT checkpoint:

```bash
python run_ideal_text.py --dataset ND0 --model_name Conv4 --noise_type OOD --noises 1 --stop_epoch 400 --text_weight 0.3 --bert_dir ./bert_features --ood_bert ./bert_features/miniImageNet_BERT_embeddings.pt
```

Change ND0 to ND1/ND2/ND3 and model to ResNet12 as needed. For Jupyter, use `%run run_ideal_text.py ...` with the same arguments.

## Methodological notes

- Frozen GIT captioner + frozen BERT mean-pooled features, no text fine-tuning.
- Original visual IDEAL `set_forward` and calibration unchanged.
- Additional text logit: `10 * cosine(query_text, mean(support_text))`, multiplied by `text_weight` and added to IDEAL's visual scores.
- The text contribution is **not** included in the intra/inter-class calibration losses; those remain image-based.
- All noise changes are mirrored in path assignments; OOD miniImageNet embeddings are mandatory.
- Separate checkpoint naming (`IDEALText` and `lambda`) prevents collisions with image-only checkpoints.
- Validation is used for checkpoint selection. Choose `text_weight` using validation data, never novel/test data.
- For rigorous paired comparisons, save/reuse identical episode manifests or seeds across methods. The current implementation seeds globally but does not yet serialize episode manifests.
- `accuracy_ci95` follows the existing normal-approximation episode CI, not a across-seed uncertainty estimate.
- The original runner imports `methods.conditionasimsiam`. If your actual module filename differs, update **only the import in `run_ideal_text.py`** to match the working original runner.

## Limitations / checks

- Not run end-to-end here because dataset files, image files, and trained checkpoints are on your server.
- All paths in BERT `.pt` files must match paths in JSON after filesystem resolution. Missing matches stop execution.
- Confirm OOD miniImageNet feature file includes **all** images in the selected `novel.json` OOD pool, not just a subset; otherwise create a filtered pool JSON and adjust runner accordingly.
- GPU memory demand is close to original IDEAL; BERT is not loaded during IDEAL training, only its cached vectors.
