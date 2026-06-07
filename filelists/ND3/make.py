import numpy as np
import os
import json
import pandas as pd
from tqdm import tqdm

 #Load the captions CSV
image_captions_file = "/home/xim/xu2202_main/IDEAL-main/filelists/ND3/ND3_captions.csv"  # Update this path
captions_df = pd.read_csv(image_captions_file)

# Convert the captions dataframe to a dictionary for easy lookup using basenames
captions_dict = {
    os.path.basename(name): caption
    for name, caption in zip(captions_df["image_names"], captions_df["image_data"])
}


cwd = os.getcwd()
base_path = os.path.join(cwd, 'ND3', 'train_images')
val_path = os.path.join(cwd, 'ND3', 'val_images')
novel_path = os.path.join(cwd, 'ND3', 'test_images')

 
base = {}
base['label_names'] = []
base['image_names'] = []
base['image_labels'] = []
base['image_data'] = []
with open('base_labels.txt') as file:
    base['label_names'] = file.readlines()
for i in tqdm(range(3)):
    names = os.listdir(os.path.join(base_path, 'C%04d' % i))
    for name in names:
        base['image_names'].append(os.path.join(base_path, 'C%04d' % i, name))
        base['image_labels'].append(i)
        caption = captions_dict.get(name, "No Caption Available")
        base['image_data'].append(caption)

val = {}
val['label_names'] = []
val['image_names'] = []
val['image_labels'] = []
val['image_data'] = []
with open('val_labels.txt') as file:
    val['label_names'] = file.readlines()
for i in tqdm(range(1)):
    names = os.listdir(os.path.join(val_path, 'C%04d' % i))
    for name in names:
        val['image_names'].append(os.path.join(val_path, 'C%04d' % i, name))
        val['image_labels'].append(i + 3)
        caption = captions_dict.get(name, "No Caption Available")
        val['image_data'].append(caption)

novel = {}
novel['label_names'] = []
novel['image_names'] = []
novel['image_labels'] = []
novel['image_data'] = []
with open('novel_labels.txt') as file:
    novel['label_names'] = file.readlines()
for i in tqdm(range(3)):
    names = os.listdir(os.path.join(novel_path, 'C%04d' % i))
    for name in names:
        novel['image_names'].append(os.path.join(novel_path, 'C%04d' % i, name))
        novel['image_labels'].append(i + 4)
        caption = captions_dict.get(name, "No Caption Available")
        novel['image_data'].append(caption)
        

json.dump(base, open('base.json', 'w'))
json.dump(val, open('val.json', 'w'))
json.dump(novel, open('novel.json', 'w'))

data = json.load(open('base.json'))
print(data.keys())
print(len(data['label_names']))
print(len(data['image_names']))
print(len(data['image_labels']), np.min(data['image_labels']), np.max(data['image_labels']))
print(len(data['image_data']))

data = json.load(open('val.json'))
print(data.keys())
print(len(data['label_names']))
print(len(data['image_names']))
print(len(data['image_labels']), np.min(data['image_labels']), np.max(data['image_labels']))
print(len(data['image_data']))

data = json.load(open('novel.json'))
print(data.keys())
print(len(data['label_names']))
print(len(data['image_names']))
print(len(data['image_labels']), np.min(data['image_labels']), np.max(data['image_labels']))
print(len(data['image_data']))
