import cv2
import os, shutil
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
from torchvision import transforms
from PIL import Image
import numpy as np
import random
import subprocess
import json
import pandas
from data.vid.kinetics_dataloader import *
from data.vid.something_dataloader import *

class AggregateDataset(Dataset):
    def __init__(self, hparams, split, transform, normal_lookup=None): #NOTE removed ucf and just doing ssv2 and kinetics
        """
        Args:
            dataset_dir (str): Directory containing class folders.
            transform (callable, optional): Optional transform to apply to frames.
            num_frames (int): Number of frames to return from each video.
        """
        self.current_user = os.getenv("USER")
        self.hparams = hparams

        self.transform = transform
        if normal_lookup:
            k400_std, k400_mean = normal_lookup['k400']
            self.k400_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.Normalize(mean=k400_mean, std=k400_std)
            ])

            ssv2_std, ssv2_mean = normal_lookup['ssv2']
            self.ssv2_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.Normalize(mean=ssv2_mean, std=ssv2_std)
            ])
        else:
            self.k400_transform = transform
            self.ssv2_transform = transform

        self.num_frames = self.hparams.context_length
        self.time_between_frames = self.hparams.time_between_frames
        self.split = split
        
        # FFPROBE SETUP
        # First, check if ffprobe is in path
        if shutil.which("ffprobe") is not None:
            self.ffprobe = shutil.which("ffprobe")
        # If not, check if environment variable was set
        elif os.getenv("FFPROBE_PATH") is not None:
            self.ffprobe = os.getenv("FFPROBE_PATH")
        # Finally, check if hparam is set
        elif hparams.ffprobe_path is not None:
            self.ffprobe = hparams.ffprobe_path
        else:
            raise ValueError("no ffprobe found")
        assert os.path.exists(self.ffprobe)
        
        self.class_names = []
        self.file_list = []
        self.labels = []
        self.frame_lookup = None
        
        
        
        if self.split not in ('train','test'):
            raise(Exception("Split must be train or test"))
            
            
        if self.split == 'test':
            self.kinetics_dataset = Kinetics400Dataset(hparams, split, transform)
            self.something_dataset = SomethingDataset(hparams, split, transform)
        elif self.split == 'train':
            self.kinetics_dataset = Kinetics400Dataset(hparams, split, self.k400_transform)
            self.something_dataset = SomethingDataset(hparams, split, self.ssv2_transform)

            self.kinetics_val_dataset = Kinetics400Dataset(hparams, 'val', self.k400_transform)
            self.something_val_dataset = SomethingDataset(hparams, 'val', self.ssv2_transform)

        self.labels = self.kinetics_dataset.labels + self.something_dataset.labels
        self.class_names = self.kinetics_dataset.class_names + self.something_dataset.class_names
        self.file_list = self.kinetics_dataset.file_list + self.something_dataset.file_list

            
    def __len__(self):
        return len(self.kinetics_dataset) + len(self.something_dataset)

    def __getitem__(self, idx):
        if idx < len(self.kinetics_dataset):
            return self.kinetics_dataset[idx]
        elif idx < len(self.kinetics_dataset)+len(self.something_dataset):
            return self.something_dataset[idx-len(self.kinetics_dataset)]
        else:
            raise Exception(f"Index {idx} out of range {len(self)}")
    
    def train_val_split(self, val_split_pct = 0.2):
        
        train_ds = ConcatDataset([self.kinetics_dataset, self.something_dataset])
        val_ds = ConcatDataset([self.kinetics_val_dataset, self.something_val_dataset])

        return train_ds, val_ds
        
    
