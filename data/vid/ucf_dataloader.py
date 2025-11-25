import cv2
import os, shutil
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import random
import subprocess
import json
import getpass
from data.vid.video_data_utils import handle_corrupt_file
import os


VAL_SPLIT = 0.2
class UCF101Dataset(Dataset):
    def __init__(self, hparams, split, transform, labels_dir = 'ucfTrainTestlist/', ignore_filepath=""):
        """
        Args:
            dataset_dir (str): Directory containing class folders.
            transform (callable, optional): transform to apply to frames.
            num_frames (int): Number of frames to return from each video.
        """

        self.current_user = getpass.getuser()
        self.hparams = hparams
        hf_home = os.getenv('HF_HOME')
        self.dataset_dir = self.hparams.dataset_dir if self.hparams.dataset_dir != "" else hf_home
        self.labels_dir = self.dataset_dir + '/' + labels_dir
        self.transform = transform
        self.num_frames = self.hparams.context_length
        self.time_between_frames = self.hparams.time_between_frames
        self.split = split
        
        # This file is the output of ebt/utils/find_corrupt_files.py
        # Filepath should be .../ebt/utils/corrupt_files/<dataset>.txt
        self.potential_ignore_filepath = '/'.join([os.getcwd(), "data/vid/corrupt_files/ucf.txt"])
        if ignore_filepath:
            self.ignore_filepath = self.labels_dir + ignore_filepath
        elif os.path.exists(self.potential_ignore_filepath):
            self.ignore_filepath = self.potential_ignore_filepath
        else:
            self.ignore_filepath = ignore_filepath
        
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
        

        self.class_names = [None] # Labels in classInd.txt are indexed at 1
        self.file_list = []
        self.labels = []
        self.frame_lookup = None        

        with open(self.labels_dir+'classInd.txt') as f:
            for row in f.read().strip().split('\n'):
                
                ind, name = row.split(' ')
                self.class_names.append(name)
        
        if self.split not in ('train','test', 'val'):
            raise(Exception("Split must be train or test"))
        if not self.hparams.debug_mode:
             
            if self.split in ('train', 'val'):
                labels_file = 'trainlist01.txt'
            elif self.split == 'test':
                labels_file ='testlist01.txt'
        else:
            labels_file = 'debuglist.txt' if self.split in ('train', 'val') else 'debuglist.txt' # small list to be quick
        
        with open(self.labels_dir+labels_file) as f:
            for row in f.read().strip().split('\n'):
                entry = row.split(' ')[0]
                name, filepath = entry.split('/')
                self.labels.append(self.class_names.index(name))
                self.file_list.append(self.dataset_dir + '/' + filepath)
        
        # self.file_list = list(filter(os.path.exists, self.file_list))
        
        if self.ignore_filepath:
            with open(self.ignore_filepath) as f:
                self.ignore_files = f.read()
            new_file_list = []
            for fp in self.file_list:
                fp_components = os.path.normpath(fp).split('/')
                new_fp = '/'.join(fp_components[-2:]) 
                if new_fp not in self.ignore_files:
                    new_file_list.append(fp)
            self.file_list = new_file_list
                
    def get_length(self, filename):
        command = f'{self.ffprobe} -v quiet -print_format json -show_format {filename}'
        data = json.loads(subprocess.check_output(command, shell=True))
        return float(data['format']['duration'])

    def get_num_frames(self, filepath):
        cap = cv2.VideoCapture(filepath)
        estimate = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        count=0
        for i in range(estimate):
            ret, img = cap.read()
            if not ret: break
            count += 1
            
        return count - 1
    
    def get_num_frames_static(self, filepath):
        if self.frame_lookup == None:
            self.frame_lookup = json.load(open(self.labels_dir+"lengths.json"))
        
        return self.frame_lookup[filepath]-1
        
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        video_path = self.file_list[idx]
        try:
            video_length = self.get_length(video_path)
            label = self.labels[idx]

            cap = cv2.VideoCapture(video_path)
            total_frames = self.get_num_frames(video_path)-1
            frame_idx_list = []

            if total_frames < self.num_frames:
                frame_idx_list = torch.linspace(0, total_frames, steps=self.num_frames).long() # just duplicate frames
            elif self.hparams.use_raw_framerate:
                if self.hparams.no_randomness_dataloader:
                    start_frame = 0
                else: # normal condition
                    start_frame = random.random() * (total_frames-self.num_frames)
                frame_idx_list = torch.arange(start_frame, start_frame+self.num_frames).long()
            elif self.hparams.sampling_rate != 0:
                required_frames = self.hparams.sampling_rate * (self.num_frames-1)
                if required_frames > total_frames:
                    # print("required_frames was > total_frames", required_frames, total_frames)
                    frame_idx_list = torch.linspace(0, total_frames, steps=self.num_frames).long() # Default case when you can't put the desired space between frames
                else:
                    start_frame = int(random.random() * (total_frames-required_frames))
                    frame_idx_list = torch.linspace(start_frame, start_frame+required_frames, steps=self.num_frames).long()
            else: # uses time_between_frames hparam
                required_length = self.time_between_frames * (self.num_frames-1)
                if required_length > video_length:
                    frame_idx_list = torch.linspace(0, total_frames, steps=self.num_frames).long() # Default case when you can't put the desired space between frames
                else:
                    start_time = random.random() * (video_length-required_length)
                    end_time = start_time + required_length

                    frame_idx_list = (total_frames/video_length
                                    *torch.linspace(start_time, end_time, steps=self.num_frames)).long()
            
            
            frames = []
            

            for i in range(self.num_frames):           
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx_list[i].item())
                ret, frame = cap.read()
                if not ret: 
                    print(f"frame is blank at {frame_idx_list[i].item()}")
                    print(video_path)

                if self.transform:
                    square_pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    transformed_frame = self.transform(square_pil_image)
                    frames.append(transformed_frame)
                else:
                    raise ValueError("Please specify transform")


            cap.release()

            frames = torch.stack(frames)
            if self.hparams.model_name in ["ebt", "baseline_transformer"]:
                return frames
            else:
                return frames, label
        except Exception as exception:
            log_path = self.ignore_filepath if self.ignore_filepath else self.potential_ignore_filepath
            handle_corrupt_file(exception, video_path, log_path)