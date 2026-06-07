# text-OFSL

## Requirements

- Python >= 3.6
- PyTorch (GPU version) >= 1.5
- NumPy >= 1.13.3
- Scikit-learn >= 0.20

## Getting started

### ND0

- Add the dataset to directory `./filelists/ND0`
- Download [ND0](https://drive.google.com/file/d/1IiwWtydp8EpNAFZzoPmjT3GBcYuThjvy/view?usp=sharing)
- Generate captions for the dataset by executing cells in `caption_generation.ipynb`
- run `%run make.py` and `%run make_all.py` in jupyter notebook

### ND1

- Add the dataset to directory `./filelists/ND1`
- Download [ND1](https://drive.google.com/file/d/1-sjK2X-nJch4gPgYkmqBfHGmBBvf-MaD/view?usp=sharing)
- Generate captions for the dataset by executing cells in `caption_generation.ipynb`
- run `%run make.py` and `%run make_all.py` in jupyter notebook

### ND2

- Add the dataset to directory `./filelists/ND2`
- Download [ND2](https://drive.google.com/file/d/1mpCKFpaxGoZtDVfqTXDpuYyvUyzcknhz/view?usp=sharing)
- Generate captions for the dataset by executing cells in `caption_generation.ipynb`
- run `%run make.py` injupyter notebook

### ND3

- Add the dataset to directory `./filelists/ND3`
- Download [ND3](https://drive.google.com/file/d/1FLEokjZvmR8cZCdhG5pXgtmj-moo4mx8/view?usp=sharing)
- Generate captions for the dataset by executing cells in `caption_generation.ipynb`
- run `%run make.py` and `%run make_all.py` in jupyter notebook

## Running the scripts

To pre-train the contrastive network in jupyter notebook, use:

```
%run run_IDEAL_pre_train.py --dataset ND0 --model_name Conv4 --train_n_way 5 --test_n_way 5 --n_shot 5 --device cuda:0
```

To train and test the IDEAL model in jupyter notebook, use:

```
%run run_IDEAL.py --dataset ND0 --noises 1 --noise_type IT --model_name Conv4 --train_n_way 5 --test_n_way 5 --n_shot 5 --device cuda:0 --meta_algorithm IDEAL --attention_method bilstm --eta 0.1 --gamma 0.1
```

## Acknowledgment

Our project references the codes in the following repo and paper.

[IDEAL](https://github.com/anyuexuan/IDEAL/blob/main/README.md?plain=1)

An Y, Xue H, Zhao X, Wang J. From instance to metric calibration: A unified framework for open-world few-shot learning. 
IEEE Transactions on Pattern Analysis and Machine Intelligence. 2023 Feb 10

Mengye Ren, Eleni Triantafillou, Sachin Ravi, Jake Snell, Kevin Swersky, Joshua B. Tenenbaum, Hugo Larochelle, Richard S. Zemel. Meta-Learning for Semi-Supervised Few-Shot Classification. ICLR 2018.
