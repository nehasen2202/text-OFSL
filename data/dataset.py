import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


import torch
from PIL import Image
import json
import numpy as np
import torchvision.transforms as transforms

from transformers import BertTokenizer, BertModel

def identity(x):
    return x
    
# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
bert_model.eval() # ✅ Important: evaluation mode (disable dropout)
bert_model.cuda()  # ✅ Important: move to GPU

class SimpleDataset:
    def __init__(self, data_file, transform, target_transform=identity):
        with open(data_file, 'r') as f:
            self.meta = json.load(f)

        self.transform = transform
        self.target_transform = target_transform

 

    def vectorize_text(self, text):
        """Vectorize text using BERT."""
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        
        inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = bert_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze()

    def __getitem__(self, index):
        # Load and process image
        image_path = os.path.join(self.meta['image_names'][index])
        img = Image.open(image_path).convert('RGB')
        img = self.transform(img)

        # Process target label
        target = self.target_transform(self.meta['image_labels'][index])

        # Vectorize image data (text) if present
        if 'image_data' in self.meta:
            text_data = self.meta['image_data'][index]
            text_vector = self.vectorize_text(text_data)
        else:
            text_vector = torch.zeros((768,), device='cuda')  # Default vector if no text data available

        return img, target, text_vector

    def __len__(self):
        return len(self.meta['image_names'])


class SetDataset:
    def __init__(self, data_file, batch_size, transform):
        with open(data_file, 'r') as f:
            self.meta = json.load(f)

        self.cl_list = np.unique(self.meta['image_labels']).tolist()

        self.sub_meta = {}
        for cl in self.cl_list:
            self.sub_meta[cl] = []

        for x, y, z in zip(self.meta['image_names'], self.meta['image_labels'], self.meta.get('image_data', [])):
            self.sub_meta[y].append((x, z))  # Include image_data (z) with image_names (x)

        self.sub_dataloader = []
        sub_data_loader_params = dict(batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=0,  # use main thread only or may receive multiple batches
                                      pin_memory=False)
        for cl in self.cl_list:
            sub_dataset = SubDataset(self.sub_meta[cl], cl, transform=transform)
            self.sub_dataloader.append(torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params))

    def __getitem__(self, index):
        return next(iter(self.sub_dataloader[index]))

    def __len__(self):
        return len(self.cl_list)
        

class SubDataset:
    def __init__(self, sub_meta, cl, transform=transforms.ToTensor(), target_transform=identity):
        self.sub_meta = sub_meta
        self.cl = cl
        self.transform = transform
        self.target_transform = target_transform

        

    def vectorize_text(self, text):
        """Vectorize text using BERT."""
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        
        inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = bert_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze()

    def __getitem__(self, index):
        # Load and process image
        image_path, image_data = self.sub_meta[index]
        img = Image.open(image_path).convert('RGB')
        img = self.transform(img)

        # Process target label
        target = self.target_transform(self.cl)

        # Process text vector
        if image_data:
            text_vector = self.vectorize_text(image_data)
        else:
            text_vector = torch.zeros((768,), device='cuda')  # Default vector if no text data available

        return img, target, text_vector

    def __len__(self):
        return len(self.sub_meta)



class EpisodicBatchSampler(object):
    def __init__(self, n_classes, n_way, n_episodes):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            yield torch.randperm(self.n_classes)[:self.n_way]

