import os
import random
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import pandas as pd
import json

AUs = ['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r']

class TestDataset(Dataset):
        def __init__(self, feature_dir, txt_path):
                self.x, self.y = [], []
                label_mapping = {}
                next_label = 0
                with open(txt_path) as f:
                        next(f)
                        for line in tqdm(f.readlines()):
                                video_name, label = line.strip().split(',')

                                if not label in label_mapping.keys():
                                        label_mapping[label] = next_label
                                        next_label += 1

                                feature = self.get_fau(feature_dir, video_name)
                                self.x.append(feature)
                                self.y.append(label_mapping[label])

        def __getitem__(self, idx):
                return self.x[idx], self.y[idx], None, None 

        def __len__(self):
                return len(self.y)

        def get_fau(self, feature_dir, video_name):
                df = pd.read_csv(os.path.join(feature_dir, f"{video_name.replace('.mp4', '')}.csv"))
                feature = torch.tensor(df[AUs].values).float() # (T, num_fau)
                return feature

class TrainDataset(Dataset):
        def __init__(self, cfg):
                self.x, self.y = [], []
                label_mapping = {}
                next_label = 0
                '''
                Fake Data
                '''
                with open(cfg["fake"]["train_txt"]) as f:
                        next(f)
                        for line in tqdm(f.readlines()):
                                # split information
                                video_name, label = line.strip().split(',')

                                # remap label
                                if not label in label_mapping.keys():
                                        label_mapping[label] = next_label
                                        next_label += 1
                                
                                feature = self.get_fau(cfg["fake"]["feature_dir"], video_name)
                                self.x.append(feature)
                                self.y.append(label_mapping[label])

                
                self.real_x = {}
                '''
                Real Data
                ''' 
                with open(cfg["real"]["train_txt"]) as f:
                        next(f)
                        for line in tqdm(f.readlines()):
                                # split information
                                video_name, label = line.strip().split(',')
                                if not label in label_mapping.keys():
                                        continue
                                
                                if not label_mapping[label] in self.real_x.keys():
                                        self.real_x[label_mapping[label]] = []
                                feature = self.get_fau(cfg["real"]["feature_dir"], video_name)
                                self.real_x[label_mapping[label]].append(feature)

        def __getitem__(self, idx):
                anchor = random.choice(self.real_x[self.y[idx]])
                fn = min(random.randint(0, len(self.x[idx])-1), random.randint(0, len(anchor)-1))
                posneg = self.x[idx][fn]
                anchor = anchor[fn]
                return self.x[idx], self.y[idx], anchor, posneg # a,p,n = E(gen), E()

        def __len__(self):
                return len(self.x)

        def get_fau(self, feature_dir, video_name):
                df = pd.read_csv(os.path.join(feature_dir, f"{video_name.replace('.mp4', '')}.csv"))
                feature = torch.tensor(df[AUs].values).float() # (T, num_fau)
                return feature