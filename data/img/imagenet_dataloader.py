from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader
import getpass
from torchvision import transforms
import os

#credit: https://huggingface.co/datasets/imagenet-1k'
#NOTE: if you are having issues with this dataloader and perms you need to add your HF token
#see these links - https://discuss.huggingface.co/t/imagenet-1k-is-not-available-in-huggingface-dataset-hub/25040 https://huggingface.co/docs/hub/security-tokens
class ImageNetDataset(Dataset):
    def __init__(self, hparams, split, transform):
        self.hparams = hparams
        self.transform = transform
        # current_user = getpass.getuser()
        split = 'validation' if split in ["valid", "val", "validate"] else split
        
        hf_home = os.getenv('HF_HOME')
        dataset_dir = self.hparams.dataset_dir if self.hparams.dataset_dir != "" else hf_home
        
        hf_token = os.getenv('HF_TOKEN')
        if not hf_token:
            print("warning: HF_TOKEN not set. you may need authentication for ImageNet access.")
        
        # print(f"loading ImageNet from cache: {dataset_dir if dataset_dir else 'HF default'}")
        
        self.ds = load_dataset(
            "ILSVRC/imagenet-1k", 
            cache_dir=dataset_dir,
            token=hf_token,
            trust_remote_code=True, 
            split=split,
            verification_mode='no_checks',
            num_proc=min(12, os.cpu_count()) 
        )
        
        print(f"ImageNet {split} dataset loaded with {len(self.ds)} samples")

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        sample = self.ds[idx]
        image = sample['image']
        image_mode = image.mode
        if image_mode == 'L':
            image = image.convert("RGB")
        elif image_mode == 'RGBA':
            image = image.convert("RGB")
        
        transformed_image = self.transform(image)
        return {'image': transformed_image, 'label': sample['label']}
