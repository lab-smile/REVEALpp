# dataloader here
import os
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import pandas as pd
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from transformers import BertTokenizer, AutoModel, AutoTokenizer, AutoConfig, EncoderDecoderModel
from omegaconf import OmegaConf
import os.path as op
import random

from utils import load_from_yaml_file, read_json, load_config_file


import torch
import numpy as np

def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=Image.BICUBIC),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.4225, 0.4012, 0.3659), (0.2681, 0.2635, 0.2763)), # COCO mean, std
    ])


def get_img_id_to_img_path(annotations):
    img_id_to_img_path = {}
    for img_info in annotations['images']:
        img_id = img_info['id']
        file_name = img_info['path']
        img_id_to_img_path[img_id] = file_name
    
    return img_id_to_img_path

def get_img_id_to_captions(annotations):
    img_id_to_captions = {}
    for caption_info in annotations['annotations']:
        img_id = caption_info['image_id']
        if img_id not in img_id_to_captions:
            img_id_to_captions[img_id] = []
        
        caption = caption_info['caption']
        img_id_to_captions[img_id].append(caption)
    
    return img_id_to_captions


def init_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained('UFNLP/gatortronS')
    tokenizer.add_special_tokens({'bos_token':'[DEC]'})
    tokenizer.add_special_tokens({'additional_special_tokens':['[ENC]']})       
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]  
    return tokenizer



class FundusDataset(torch.utils.data.Dataset):
    def __init__(self, config, context_length=77, input_resolution=224):
        self.config = config

        annotation_file = self.config.train_annotation_file
        annotations = read_json(annotation_file)

        self.img_id_to_filename = get_img_id_to_img_path(annotations)
        # print("img_id_to_filename : ", self.img_id_to_filename)

        self.img_id_to_captions = get_img_id_to_captions(annotations)

        self.img_ids = list(self.img_id_to_filename.keys())
        # print("total image ids = ", len(self.img_ids))

        self.img_dir = annotations['images']
        # print("img dir : ", self.img_dir)

        self.transform = _transform(input_resolution)
        self.context_length = context_length

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]

        # randomly pick one caption from the image captions
        text_input = random.choice(self.img_id_to_captions[img_id])

        img_filename = self.img_id_to_filename[img_id]

        #img_path = op.join(self.img_dir, img_filename)
        img_path = img_filename
        img = Image.open(img_path)
        img_input = self.transform(img)
        
        return img_input, text_input
    
                
            
class FundusDataset_Fewshot(torch.utils.data.Dataset):
    def __init__(self, config, input_resolution=224):
        self.config = config

        annotation_file = self.config.train_annotation_file
        annotations = read_json(annotation_file)

        self.img_id_to_filename = get_img_id_to_img_path(annotations)
        # print("img_id_to_filename : ", self.img_id_to_filename)

        self.img_id_to_captions = get_img_id_to_captions(annotations)

        self.img_ids = list(self.img_id_to_filename.keys())
        # print("total image ids = ", len(self.img_ids))

        self.img_dir = annotations['images']
        # print("img dir : ", self.img_dir)
             
        AD_data_dir = './data/UKB/AD.csv'
        convert_dict = {'eid': str}
        df = pd.read_csv(AD_data_dir)
        df = df.astype(convert_dict)
        
        self.AD_list =  df['eid'].tolist()

        self.transform = _transform(input_resolution)

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]

        img_filename = self.img_id_to_filename[img_id]

        #img_path = op.join(self.img_dir, img_filename)
        img_path = img_filename
        img = Image.open(img_path)
        img_input = self.transform(img)
        
        if str(img_id) in self.AD_list:
            label=torch.tensor(1)
        else:
            label=torch.tensor(0)
        
        return img_input, label
    
    def AD_classes(self):  
        output = []       
        for i in self.img_ids:
            if str(i) in self.AD_list:
                diagnosis = 1
            else:
                diagnosis = 0 
            output.append((i, diagnosis))
            
        return(output)

    
class FundusDataset_Fewshot_Multimodal(torch.utils.data.Dataset):
    def __init__(self, config, tokenizer, input_resolution=224):
        self.config = config
        self.tokenizer = tokenizer

        annotation_file = self.config.train_annotation_file
        annotations = read_json(annotation_file)

        self.img_id_to_filename = get_img_id_to_img_path(annotations)
        # print("img_id_to_filename : ", self.img_id_to_filename)

        self.img_id_to_captions = get_img_id_to_captions(annotations)

        self.img_ids = list(self.img_id_to_filename.keys())
        # print("total image ids = ", len(self.img_ids))

        self.img_dir = annotations['images']
        # print("img dir : ", self.img_dir)
             
        AD_data_dir = './data/UKB/AD.csv'
        convert_dict = {'eid': str}
        df = pd.read_csv(AD_data_dir)
        df = df.astype(convert_dict)
        
        self.AD_list =  df['eid'].tolist()
        self.transform = _transform(input_resolution)

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        
        text_input = [random.choice(self.img_id_to_captions[img_id])]
        img_filename = self.img_id_to_filename[img_id]

        #img_path = op.join(self.img_dir, img_filename)
        img_path = img_filename
        img = Image.open(img_path)
        img_input = self.transform(img)
        
        if str(img_id) in self.AD_list:
            label=torch.tensor(1)
        else:
            label=torch.tensor(0)
        
        return img_input, text_input, label
    
    def AD_classes(self):  
        output = []       
        for i in self.img_ids:
            if str(i) in self.AD_list:
                diagnosis = 1
            else:
                diagnosis = 0 
            output.append((i, diagnosis))
            
        return(output)
    

class VQADataset(torch.utils.data.Dataset):
    def __init__(self, json):
        self.img_path = json['image_id']
        self.question = json['question']
        self.label = json['label']
        
        self.transform = _transform(224)
        #self.tokenizer = _tokenizer('gatortron')


    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_path[idx])
        image = Image.open(img_path)
        label = self.label[idx]
        question = self.question[idx]
        
        image = self.transform(image)

        if len(label['ids']) == 1:
            label = label['ids'][0]
        else:
            label = label['ids'][label['weights'].index(1)]
            
        return image, question, label