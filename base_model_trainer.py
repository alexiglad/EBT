import pytorch_lightning as L
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms
from torchvision.transforms import ToPILImage
from torch.distributed import all_reduce
import wandb
import gc

from data.vid.ucf_dataloader import *
from data.vid.kinetics_dataloader import *
from data.img.imagenet_dataloader import *
from data.img.coco_tiny_dataset import COCOTinyDataset
from data.img.coco_medium_dataset import COCOMediumDataset
from data.vid.aggregate_dataloader import *
from data.vid.vid_synthetic_dataset import VIDSyntheticDataset
from data.nlp.pajama_dataloader import RedPajamaDataset
from data.nlp.fineweb_dataloader import FineWebDataset
from data.nlp.collator import NLP_HF_Collator
from data.nlp.bigbench_dataloader import BigBenchDataset
from data.nlp.gsm8k_dataloader import GSM8KDataset
from data.nlp.lambada_dataset import LambadaDataset
from data.nlp.squad_dataloader import SQuADDataset
from data.nlp.ai2arc_dataloader import AI2ArcDataset
from data.nlp.planbench_dataloader import PlanBenchDataset
from data.nlp.synthetic_dataset import NLPSyntheticDataset

from model.vid.ebt import EBT_VID
from model.nlp.ebt import EBT_NLP
from model.img.ebt_t2i import EBT_IMG_T2I
from model.img.ebt_denoise import EBT_IMG_Denoise

from model.vid.baseline_transformer import Baseline_Transformer_VID
from model.nlp.baseline_transformer import Baseline_Transformer_NLP

from model.img.dit_t2i import Diffusion_Transformer_IMG_T2I
from model.img.dit_denoise import Diffusion_Transformer_IMG_Denoise

from model.model_utils import save_frames, denormalize, load_image_encoder, center_crop_arr
from inference.nlp.generate_text import generate_text, get_ppl
from inference.vid.generate_video import generate_video
from inference.img.generate_image import generate_image
from optimization import (WarmUpCosineAnnealingLR, LARS, exclude_bias_and_norm, StableAdamW, StableAdamWUnfused)
from utils import text_logger
from utils.metrics_calculator import get_torchmetrics


class ModelTrainer(L.LightningModule):
    def __init__(self, hparams, trained_model = None):
        super().__init__()
        if isinstance(hparams, dict):#passed in from model ckpt
            self.hparams.update(hparams)
        else:
            self.hparams.update(vars(hparams))
        # self.txt_logger = hparams.txt_logger if txt_logger == None else txt_logger # txt_logger is no longer supported
        if self.hparams.modality == "NLP":
            if "execution_mode" in self.hparams and "save_generation_logs_dir" in self.hparams and self.hparams.execution_mode == "inference": # two of these are sanity check for loading pretrained ckpt that may not have newer params
                print("setting up infer logger")
                self.infer_logger = text_logger.setup_jsonl_logger(log_filename = "results.jsonl", base_log_dir=self.hparams.save_generation_logs_dir)
        if self.hparams.modality == "VID": #is computer vision
            self.image_dims = self.hparams.image_dims # list size two
            self.num_generated_videos = 0
            if self.hparams.custom_image_normalization:
                self.transform = transforms.Compose([
                    transforms.Resize((self.image_dims[0], self.image_dims[1])),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])

                normal_lookup = { #NOTE is std, mean
                    "ucf101": ([1.04731617, 1.04372056, 1.02795228], [-0.40689788, -0.36098219, -0.25687788]),
                    "k400": ([1.00370078, 0.99871626, 0.97407404], [-0.24295556, -0.24931058, -0.13959686]),
                    "smth": ([0.90832217, 0.93885971, 0.93745849], [-0.06761328, -0.12692231, -0.01916805]),
                    "ImageNet": ([1, 1, 1], [0, 0, 0])
                }

                normal_lookup["something"] = normal_lookup["smth"]
                normal_lookup["ImageNet1k"] = normal_lookup["ImageNet"]
                self.normal_lookup = normal_lookup

                if self.hparams.dataset_name in normal_lookup:
                    std, mean = normal_lookup[self.hparams.dataset_name]
                    self.transform.transforms.append(transforms.Normalize(mean=mean, std=std))
                elif self.hparams.dataset_name in ["aggregate"]: # these are combined datasets
                    pass
                else:
                    raise ValueError(f"{self.hparams.dataset_name} not in normal lookup")
                    
            else:
                if self.hparams.vae_normalization:
                    self.transform = transforms.Compose([
                        transforms.Resize((self.image_dims[0], self.image_dims[1])),
                        transforms.ToTensor(),
                        transforms.Normalize([0.5], [0.5])
                    ])
                else: # imagenet standardization
                    self.transform = transforms.Compose([
                        transforms.Resize((self.image_dims[0], self.image_dims[1])),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])
            self.reset_image_encoder_decoder = False
        if self.hparams.modality == "IMG": # using transform from DiT codebase https://github.com/facebookresearch/DiT
            self.transform = transforms.Compose([
                transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, self.hparams.image_dims[0])),
                # transforms.RandomHorizontalFlip(), # remove this since adds more modes
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
            ])

        self.to_pil = ToPILImage()
        self.full_ds = None
        
        if trained_model is not None:
            self.model = trained_model
        else:
            if self.hparams.model_name == "ebt":
                if self.hparams.modality == "VID":
                    self.model = EBT_VID(self.hparams)
                elif self.hparams.modality == "NLP":
                    self.model = EBT_NLP(self.hparams)
                elif self.hparams.modality == "IMG": # these are bidirectional not AR
                    if self.hparams.image_task == "t2i":
                        self.model = EBT_IMG_T2I(self.hparams) 
                    elif self.hparams.image_task == "denoising":
                        self.model = EBT_IMG_Denoise(self.hparams)
                    else:
                        raise ValueError(f"task type: {self.hparams.image_task} not supported in base model trainer as a model as of now")
                else:
                    raise ValueError(f"Modality: {self.hparams.modality} not supported as a base model trainer model as of now")
            elif self.hparams.model_name == "baseline_transformer":
                if self.hparams.modality == "VID":
                    self.model = Baseline_Transformer_VID(self.hparams)
                elif self.hparams.modality == "NLP":
                    self.model = Baseline_Transformer_NLP(self.hparams)
                else:
                    raise ValueError(f"Modality: {self.hparams.modality} not supported as a base model trainer model as of now")
            elif self.hparams.model_name == "dit":
                if self.hparams.modality == "IMG":
                    if self.hparams.image_task == "t2i":
                        self.model = Diffusion_Transformer_IMG_T2I(self.hparams) # this is bidirectional not AR
                    elif self.hparams.image_task == "denoising":
                        self.model = Diffusion_Transformer_IMG_Denoise(self.hparams) # this is bidirectional not AR
                else:
                    raise ValueError(f"Modality: {self.hparams.modality} not supported as a base model trainer model as of now")
            else:
                raise ValueError(f"do not recognize model name: {self.hparams.model_name}")
        
        if self.hparams.compile_model:
            self.model = torch.compile(self.model, fullgraph=True)

        phases = ['train', 'valid', 'test']
        self.torchmetrics_dict = nn.ModuleDict()
        self.metrics = []
        for metric in self.hparams.metrics_list:
            self.metrics.append(metric)
        if len(self.metrics) > 0:
            assert self.hparams.num_classes != -1, "please set num_classes to the appropriate amount for the in use metrics. if are using accuracy and num_classes varies just set it to something that makes sense (shouldnt matter in that case)"
            assert self.hparams.metrics_task != "", "please set metrics_task to the appropriate value for your metrics"
        for phase in phases:
            for metric in self.metrics:
                self.torchmetrics_dict[f"{phase}_{metric}"] = get_torchmetrics(metric, self.hparams.metrics_average_type, self.hparams.num_classes, self.hparams.metrics_task)

        if self.hparams.wandb_watch:
            for name, module in self.model.named_modules(): # for activation logging
                module.name = name

        # Initialize step-based tracking for accurate perplexity calculation
        if self.hparams.modality == "NLP":
            self.step_stats = {
                "train": {"total_loss": 0.0, "total_tokens": 0, "step_count": 0},
                "valid": {"total_loss": 0.0, "total_tokens": 0, "step_count": 0}, 
                "test": {"total_loss": 0.0, "total_tokens": 0, "step_count": 0}
            }

        
    def on_train_start(self):
        if self.hparams.debug_unused_parameters: 
            for name, param in self.model.named_parameters():
                if param.requires_grad and "image_encoder" not in name: # NOTE need to modify this code to exclude specific frozen portions
                    print(f"registering param - {name}")
                    param.register_hook(self.create_hook(name))
                else:
                    self.model.parameters_not_to_check.add(name)

    def create_hook(self, name): #this is only used for debugging with `debug_unused_parameters`
        def hook(grad):
            self.model.used_parameters.add(name)  # Adjusted to self.model.used_parameters
        return hook
    
    @staticmethod
    def wandb_activation_hook(run, step):
        """ Weights & Biases histogram activation hook. """
        def hook(module, input, output):
            if isinstance(output, tuple):
                pass # when tried to do things had bug AttributeError: 'list' object has no attribute 'detach'
            else:
                run.experiment.log(
                    {f"activations/{module.name}": wandb.Histogram(output.detach().cpu().float())}, 
                    step=step
                )

        return hook
    
    def training_step(self, batch, batch_idx):
        if not self.hparams.no_wandb and self.hparams.wandb_watch and self.global_step % self.hparams.wandb_watch_log_freq == 0: # activation logging
            hook_handles = []
            hook_function = self.wandb_activation_hook(run=self.logger, step=self.global_step)
            for module in self.model.modules():
                if any(param.requires_grad for param in module.parameters(recurse=False)): # only do for unfrozen params that are training
                    handle = module.register_forward_hook(hook_function)
                    hook_handles.append(handle)
            
            eval_step_dict = self.eval_step(batch, "train")
            for handle in hook_handles:
                handle.remove()

        else:
            eval_step_dict = self.eval_step(batch, "train")
        
        self.log_metrics(eval_step_dict, "train")
        return eval_step_dict['loss']   
    
    def on_after_backward(self):
        if self.hparams.log_gradients:
            total_norm = 0.0
            num_parameters = 0
            num_grads_exceeding_clip_val = 0
            total_gradients = 0 # this is different from num_parameters since .parameters is for tensors of params but doesnt count each invididual parameter
            for param in self.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm  # Add the norm value to the total sum
                    num_parameters += 1
                    
                    total_gradients += torch.numel(param.grad)
                    num_grads_exceeding_clip_val += torch.sum(param.grad.abs() > self.hparams.gradient_clip_val)
                    
            assert num_parameters > 0, "no gradients after backwards detected please investigate"
            average_norm = (total_norm / num_parameters).detach()
            percentage_clipped = ((num_grads_exceeding_clip_val / total_gradients) * 100).detach()              
            
            things_to_log = {} 
            things_to_log['avg_gradient_norms'] = average_norm
            things_to_log['pct_gradient_clipped'] = percentage_clipped
            self.log_metrics(things_to_log, "train", log_torchmetrics = False)
        
    def on_train_batch_end(self, outputs, batch, batch_idx):
        #NOTE when using this may need to explicitly add code like 'if "image_encoder" not in name' for frozen params (with requires_grad == False)
        if self.hparams.debug_unused_parameters:
            all_parameters = {name for name, _ in self.model.named_parameters()}
            unused_parameters = all_parameters - self.model.used_parameters - self.model.parameters_not_to_check
            
            print(f"number of parameters total: {len(all_parameters)}")
            print(f"number of unused_parameters: {len(unused_parameters)}")
            print(f"Unused parameters: {unused_parameters}")
            print(f"Used parameters: {self.model.used_parameters}")
        
        if self.hparams.manual_gc_collect_every_n_steps != -1:
            if self.global_step % self.hparams.manual_gc_collect_every_n_steps == 0:
                print("calling GC manually")
                gc.collect()            
           
    def on_train_epoch_end(self):
        if self.hparams.optimizer != "adamw": # e.g. for lars need to manually update epoch
            optimizer = self.trainer.optimizers[0]
            optimizer.update_epoch(self.current_epoch)   

    def validation_step(self, batch, batch_idx):
        eval_step_dict = self.eval_step(batch, "valid")
        self.log_metrics(eval_step_dict, "valid")

    def test_step(self, batch, batch_idx):
        if self.hparams.execution_mode == "inference":  
            if self.hparams.modality == "NLP":
                outputs = generate_text(self.model, batch, self.hparams)
                for output in outputs:
                    self.infer_logger.log_data(output)
            elif self.hparams.modality == "VID":
                if not self.reset_image_encoder_decoder: # this is done to prevent bug where loading ckpt image encoder doesnt work well, not sure why ckpt image decoder doesnt load well, maybe related to HF
                    self.model.image_encoder = load_image_encoder(self.hparams.backbone_type, self.hparams.vit_backbone_size).to(self.device)
                    self.model.image_encoder.eval()
                    self.reset_image_encoder_decoder = True

                outputs = generate_video(self.model, batch, self.hparams, decode_frames = self.hparams.infer_generate_video) # outputs['video'] has shame shape as batch: B, S, C, W, H

                if self.hparams.infer_generate_video:
                    denormalized_predicted_videos = denormalize(outputs['video'], self.hparams.dataset_name, self.device, self.hparams.custom_image_normalization, self.hparams.vae_normalization)
                    denormalized_batch = denormalize(batch, self.hparams.dataset_name, self.device, self.hparams.custom_image_normalization, self.hparams.vae_normalization)
                    batch_size = outputs['video'].shape[0]
                    if self.trainer.world_size > 1:
                        batch_size_tensor = torch.tensor(batch_size, device=self.device)
                        all_reduce(batch_size_tensor)
                        total_batch_size = batch_size_tensor.item()
                        video_start_idx = self.num_generated_videos + (self.global_rank * batch_size)
                    else:
                        total_batch_size = batch_size
                        video_start_idx = self.num_generated_videos

                    save_frames(denormalized_predicted_videos, self.hparams.save_generation_logs_dir, 'fake', video_start_idx) 
                    save_frames(denormalized_batch, self.hparams.save_generation_logs_dir, 'real', video_start_idx)
                    if self.hparams.debug_videos:
                        save_frames(denormalized_predicted_videos[0].unsqueeze(dim=0), self.hparams.save_generation_logs_dir, 'debug', video_start_idx)
                    
                    self.num_generated_videos += total_batch_size
                outputs.pop('video')
                self.log_metrics(outputs, "test")
            elif self.hparams.modality == "IMG":
                outputs = generate_image(self.model, batch, self.hparams)
                self.log_metrics(outputs, "test")
            else:
                raise NotImplementedError(f"Inference mode not supported for modality {self.hparams.modality} yet")
        else: # all other modes just get metrics
            if self.hparams.modality == "NLP" and self.hparams.model_name == "ebt" and self.hparams.infer_ebt_advanced: # special case where we dont want to use inference mode but still use ebt advanced inference to get log ppl, energies, etc (that way dont need to generate text 1 by 1)
                outputs = get_ppl(self.model, batch, self.hparams)
                self.log_metrics(outputs, "test")
            else:
                eval_step_dict = self.eval_step(batch, "test")
                self.log_metrics(eval_step_dict, "test")
            
    def eval_step(self, batch, phase):
        things_to_log = self.model.forward_loss_wrapper(batch, phase) # things_to_log will be a dict of various things being logged. it NEEDS TO contain the 'loss' key as this is used to backprop

        if len(self.metrics) > 0:
            raise NotImplementedError("Need to implement torchmetrics stuff, i.e. looping through self.torchmetrics_dict.keys(), checking to make sure 'phase in key', and updating based off predicted and labels i.e. self.torchmetrics_dict[key].update(logits, labels), more info https://lightning.ai/docs/torchmetrics/stable/pages/lightning.html (just be careful make sure to detach logits before using them and only update current phase). recommended to possibly return things_to_log and logits from forward_loss_wrapper to do this easily")

        return things_to_log
    
    def forward(self, batch):
        return self.model(batch)

    def configure_optimizers(self): # this is a PL hook that returns optimizer and lr scheduler
        if self.hparams.modality == "NLP":
            return self.configure_optimizers_nlp()
        elif self.hparams.modality == "VID":
            return self.configure_optimizers_vid()
        elif self.hparams.modality == "IMG":
            return self.configure_optimizers_img()
        else:
            raise NotImplementedError(f"Modality {self.hparams.modality} does not have configure optimizers supported yet")
        
    def get_optimizer(self, optimizer_parameters): # function for once gotten optimizer_parameters to get optimizer, i.e. adamw, lars, etc
        if self.hparams.optimizer == "lars":
            lars_exclude_bias_and_norm = None if not self.hparams.lars_exclude_bias_bn_wd else exclude_bias_and_norm
            optimizer = LARS(optimizer_parameters, lr=self.hparams.peak_learning_rate, weight_decay=self.hparams.weight_decay, momentum=self.hparams.beta1, eta=self.hparams.lars_trust_coeff, weight_decay_filter=lars_exclude_bias_and_norm, lars_adaptation_filter=lars_exclude_bias_and_norm)
        elif self.hparams.optimizer == "stableadamw":
            optimizer = StableAdamWUnfused(optimizer_parameters, betas=[self.hparams.beta1, self.hparams.beta2])
        else:
            optimizer = torch.optim.AdamW(optimizer_parameters, betas=[self.hparams.beta1, self.hparams.beta2])
        return optimizer
    
    def on_warm_up_finished(self):
        if hasattr(self.model, 'warm_up_finished'):
            self.model.warm_up_finished()
            print("Warm up finished, calling self.model.warm_up_finished()")
        else:
            print("Warm up finished, no self.model.warm_up_finished() exists so not doing anything")
    
    def get_lr_scheduler(self, optimizer):
        cosine_annealing_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.max_scheduling_steps - self.hparams.warm_up_steps, eta_min=self.hparams.peak_learning_rate / self.hparams.min_lr_scale)
        lr_scheduler = WarmUpCosineAnnealingLR(optimizer, warm_up_steps = self.hparams.warm_up_steps, warm_up_base_lr_divider = self.hparams.warm_up_base_lr_divider, cosine_scheduler=cosine_annealing_scheduler, warm_up_finished_func=self.on_warm_up_finished)
        return lr_scheduler
    
    def get_optimizer_scheduler_dict(self, optimizer_parameters):
        optimizer = self.get_optimizer(optimizer_parameters)
        lr_scheduler = self.get_lr_scheduler(optimizer)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': lr_scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }
            
    def configure_optimizers_nlp(self):
        if self.hparams.model_name == "ebt":
            alpha_param = self.model.alpha
            other_params = [param for name, param in self.model.named_parameters() if not any(keyword in name for keyword in ['alpha'])]
            assert len(other_params) > 1, "Could not gather model params correctly please investigate"

            optimizer_parameters = [
                {'params': alpha_param, 'weight_decay': 0.0, 'lr': self.hparams.mcmc_step_size_lr_multiplier*self.hparams.peak_learning_rate},  # No weight decay for alpha
                {'params': other_params, 'weight_decay': self.hparams.weight_decay, 'lr': self.hparams.peak_learning_rate}  # Weight decay for other parameters
            ]
            return self.get_optimizer_scheduler_dict(optimizer_parameters)
            
        elif self.hparams.model_name == "baseline_transformer":
            all_params = [param for _, param in self.model.named_parameters()]
            optimizer_parameters = [
                {'params': all_params, 'weight_decay': self.hparams.weight_decay, 'lr': self.hparams.peak_learning_rate}  # Weight decay for other parameters
            ]
            return self.get_optimizer_scheduler_dict(optimizer_parameters)
        
        else:
            raise NotImplementedError(f"havent implemented configure optimizers for model {self.hparams.model_name}")

        
    def configure_optimizers_vid(self):
        if self.hparams.model_name == "ebt":
            alpha_param = self.model.alpha
            encoder_params = list(self.model.image_encoder.parameters())
            other_params = [param for name, param in self.model.named_parameters() if not any(keyword in name for keyword in ['alpha', 'image_encoder'])]
            assert len(other_params) > 1, "Could not gather model params correctly please investigate"
            
            optimizer_parameters = [
                {'params': alpha_param, 'weight_decay': 0.0, 'lr': self.hparams.mcmc_step_size_lr_multiplier*self.hparams.peak_learning_rate},  # No weight decay for alpha
                {'params': encoder_params, 'weight_decay': 0.0, 'lr': 0.0},
                {'params': other_params, 'weight_decay': self.hparams.weight_decay, 'lr': self.hparams.peak_learning_rate}  # Weight decay for other parameters
            ]
            return self.get_optimizer_scheduler_dict(optimizer_parameters)
            
        elif self.hparams.model_name == "baseline_transformer":
            encoder_params = list(self.model.image_encoder.parameters())
            other_params = [param for name, param in self.model.named_parameters() if not any(keyword in name for keyword in ['image_encoder'])]

            optimizer_parameters = [
                {'params': encoder_params, 'weight_decay': 0, 'lr': 0},
                {'params': other_params, 'weight_decay': self.hparams.weight_decay, 'lr': self.hparams.peak_learning_rate}  # Weight decay for other parameters
            ]
            return self.get_optimizer_scheduler_dict(optimizer_parameters)
        
        else:
            raise NotImplementedError(f"havent implemented configure optimizers for model {self.hparams.model_name}")
        
    def configure_optimizers_img(self):
        if self.hparams.model_name == "ebt":
            alpha_param = self.model.alpha
            other_params = [param for name, param in self.model.named_parameters() if not any(keyword in name for keyword in ['alpha', 'image_encoder', 'text_encoder'])]
            assert len(other_params) > 1, "Could not gather model params correctly please investigate"
            
            optimizer_parameters = [
                {'params': alpha_param, 'weight_decay': 0.0, 'lr': self.hparams.mcmc_step_size_lr_multiplier*self.hparams.peak_learning_rate},  # No weight decay for alpha
                {'params': other_params, 'weight_decay': self.hparams.weight_decay, 'lr': self.hparams.peak_learning_rate} # Weight decay for other parameters
            ]
            
            if self.hparams.image_task == "t2i": # do this bc other models wont have these 'sub' models
                image_encoder_params = list(self.model.image_encoder.parameters())
                optimizer_parameters.insert(1, {'params': image_encoder_params, 'weight_decay': 0, 'lr': 0})
                text_encoder_params = list(self.model.text_encoder.parameters())
                optimizer_parameters.insert(2, {'params': text_encoder_params, 'weight_decay': 0, 'lr': 0})
            
            return self.get_optimizer_scheduler_dict(optimizer_parameters)
            
        elif self.hparams.model_name == "dit":
            other_params = [param for name, param in self.model.named_parameters() if not any(keyword in name for keyword in ['image_encoder', 'text_encoder'])]

            optimizer_parameters = [
                {'params': other_params, 'weight_decay': self.hparams.weight_decay, 'lr': self.hparams.peak_learning_rate}  # Weight decay for other parameters
            ]
            if self.hparams.image_task == "t2i":
                image_encoder_params = list(self.model.image_encoder.parameters())
                optimizer_parameters.insert(0, {'params': image_encoder_params, 'weight_decay': 0, 'lr': 0})
                text_encoder_params = list(self.model.text_encoder.parameters())
                optimizer_parameters.insert(1, {'params': text_encoder_params, 'weight_decay': 0, 'lr': 0})
            
            return self.get_optimizer_scheduler_dict(optimizer_parameters)
        
        else:
            raise NotImplementedError(f"havent implemented configure optimizers for model {self.hparams.model_name}")


    def create_full_ds(self):
        if self.hparams.dataset_name == "coco_tiny":
            self.full_ds = COCOTinyDataset(self.hparams, split = "train", transform = self.transform)
        if self.hparams.dataset_name == "ucf101":
            self.full_ds = UCF101Dataset(self.hparams, split = "train", transform = self.transform)
        elif self.hparams.dataset_name == "vid_synthetic":
            self.full_ds = VIDSyntheticDataset(self.hparams)
        elif self.hparams.dataset_name == "pajama":
            self.full_ds = RedPajamaDataset(self.hparams)
        elif self.hparams.dataset_name == 'fineweb':
            self.full_ds = FineWebDataset(self.hparams)
        elif "bigbench" in self.hparams.dataset_name:
            x = self.hparams.dataset_name
            self.full_ds = BigBenchDataset(self.hparams, "train", x[x.find('_') + 1 :])
        elif self.hparams.dataset_name == "planbench":
            self.full_ds = PlanBenchDataset(self.hparams, split = "train")
        elif self.hparams.dataset_name == "nlp_synthetic":
            self.full_ds = NLPSyntheticDataset(self.hparams)
        elif self.hparams.dataset_name == "aggregate": # aggregate VID dataset combining ssv2 and k400
            self.full_ds = AggregateDataset(self.hparams, split = "train", transform = self.transform, normal_lookup=self.normal_lookup)
        else:
            raise NotImplementedError(f"haven't implemented dataset {self.hparams.dataset_name} full_ds yet")

    def setup(self, stage=None):
        # NOTE when passing stage into datasets/dataloaders use string rep not the stage param from this func since is a PL enum
        # Assign train/val datasets for use in dataloaders 
        assert self.hparams.test_split_pct == 0, "Haven't implemented nonzero value for test_split_pct yet"

        if stage == "fit":
            # all of these conditions need to have manual split
            if self.hparams.dataset_name in ["coco_tiny", "ucf101", "vid_synthetic", "pajama", "fineweb", "bigbench", "planbench", "nlp_synthetic"]:
                self.create_full_ds()
                train_samples = int(len(self.full_ds) * (1 - self.hparams.validation_split_pct))
                valid_samples = len(self.full_ds) - train_samples
                self.train_ds, self.val_ds = random_split(self.full_ds, [train_samples, valid_samples])
            elif self.hparams.dataset_name == "aggregate":
                self.create_full_ds()
                self.train_ds, self.val_ds = self.full_ds.train_val_split(val_split_pct = self.hparams.validation_split_pct)
            elif self.hparams.dataset_name == 'k400':
                self.train_ds = Kinetics400Dataset(self.hparams, split = 'train', transform = self.transform)
                self.val_ds = Kinetics400Dataset(self.hparams, split = 'val', transform = self.transform)
            elif self.hparams.dataset_name in ('something' , 'smth'):
                self.train_ds = SomethingDataset(self.hparams, split = 'train', transform = self.transform)
                self.val_ds = SomethingDataset(self.hparams, split = 'val', transform = self.transform)
            elif self.hparams.dataset_name in ('imagenet' , 'imagenet1k'):
                self.train_ds = ImageNetDataset(self.hparams, split = 'train', transform = self.transform)
                self.val_ds = ImageNetDataset(self.hparams, split = 'val', transform = self.transform)
            elif self.hparams.dataset_name == 'coco_medium':
                self.train_ds = COCOMediumDataset(self.hparams, split = "train", transform = self.transform)
                self.val_ds = COCOMediumDataset(self.hparams, split = "validation", transform = self.transform)
            elif self.hparams.dataset_name == "gsm8k":
                self.train_ds = GSM8KDataset(self.hparams, split = "train")
                self.val_ds = GSM8KDataset(self.hparams, split = "test") # no val just test https://huggingface.co/datasets/openai/gsm8k
            elif self.hparams.dataset_name == "ai2arc":
                self.train_ds = AI2ArcDataset(self.hparams, split = 'train')
                self.val_ds = AI2ArcDataset(self.hparams, split = 'validation')
            elif self.hparams.dataset_name == "squad":
                self.train_ds = SQuADDataset(self.hparams, split = 'train')
                self.val_ds = SQuADDataset(self.hparams, split = 'validation')
            else:
                raise NotImplementedError("Haven't implemented this dataset yet")
            print(f"{self.hparams.dataset_name} length of train_dataset: {len(self.train_ds)} and val_dataset: {len(self.val_ds)}")
            
        # Assign test dataset for use in dataloader(s)
        elif stage == "test":
            if self.hparams.dataset_name == "ucf101":
                self.test_ds = UCF101Dataset(self.hparams, split = "test", transform = self.transform)
            elif self.hparams.dataset_name in ('kinetics400' , 'k400'):
                self.test_ds = Kinetics400Dataset(self.hparams, split = "test", transform = self.transform)
            elif self.hparams.dataset_name in ('something' , 'smth'):
                self.test_ds = SomethingDataset(self.hparams, split = "test", transform = self.transform)
            elif self.hparams.dataset_name in ('imagenet' , 'imagenet1k'):
                self.test_ds = ImageNetDataset(self.hparams, split = "test", transform = self.transform)
            elif self.hparams.dataset_name == 'aggregate':
                self.test_ds = AggregateDataset(self.hparams, split = "test", transform = self.transform)
            elif self.hparams.dataset_name == "coco_tiny":
                self.test_ds = COCOTinyDataset(self.hparams, split = "validation", transform = self.transform) # use validation since there is no test split, splitted train into val
            elif self.hparams.dataset_name == "coco_medium":
                self.test_ds = COCOMediumDataset(self.hparams, split = "test", transform = self.transform)
            elif self.hparams.dataset_name == "pajama": # for now am assuming test split == val split, so dont save train or full ds here, just to get val split
                full_ds = RedPajamaDataset(self.hparams)
                train_samples = int(len(full_ds) * (1 - self.hparams.validation_split_pct))
                test_samples = len(full_ds) - train_samples
                _, self.test_ds = random_split(full_ds, [train_samples, test_samples])
            elif self.hparams.dataset_name == "fineweb":
                raise NotImplementedError(f"haven't implemented fineweb dataset test split yet")
            elif "bigbench" in self.hparams.dataset_name:
                x = self.hparams.dataset_name
                self.test_ds = BigBenchDataset(self.hparams, "validation", x[x.find('_') + 1 :]) #use val for testing as Bigbench only has train/val
            elif self.hparams.dataset_name == "gsm8k":
                self.test_ds = GSM8KDataset(self.hparams, split="test")
            elif self.hparams.dataset_name == "lambada":
                self.test_ds = LambadaDataset(self.hparams, split="test")
            elif self.hparams.dataset_name == "squad":
                self.test_ds = SQuADDataset(self.hparams, split="validation") # no test split use val
            elif self.hparams.dataset_name == "planbench":
                raise NotImplementedError(f"no planbench test split")
            elif self.hparams.dataset_name == "ai2arc":
                self.test_ds = AI2ArcDataset(self.hparams, split = "test")
            else:
                raise NotImplementedError("haven't implemented this dataset yet")
            print(f"{self.hparams.dataset_name} length of test_ds: {len(self.test_ds)}")
        else:
            raise ValueError(f"Unknown stage: {stage}, please investigate")
    
    def get_collate_fn(self):
        collate_fn = None if not self.hparams.modality == "NLP" else NLP_HF_Collator(self.hparams) #NOTE this assumes all modalities except NLP DONT have collator, may not be true in the future
        if self.hparams.dataset_name == "nlp_synthetic": #NOTE this is a hack to get around the fact that synthetic dataset cant return real text and thus cant use collate_fn
            collate_fn = None
        return collate_fn
    
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.hparams.batch_size_per_device, num_workers=self.hparams.num_workers, persistent_workers=True, collate_fn = self.get_collate_fn(), pin_memory = True, drop_last = False, shuffle = not self.hparams.no_shuffle, prefetch_factor=self.hparams.prefetch_factor)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.hparams.batch_size_per_device, num_workers=self.hparams.num_workers, persistent_workers=True, collate_fn = self.get_collate_fn(), pin_memory = True, drop_last = False, shuffle = False, prefetch_factor=self.hparams.prefetch_factor)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.hparams.batch_size_per_device, num_workers=self.hparams.num_workers, persistent_workers=True, collate_fn = self.get_collate_fn(), pin_memory = True, drop_last = False, shuffle = False, prefetch_factor=self.hparams.prefetch_factor)

    def log_metrics(self, metrics_dict, phase, log_torchmetrics = True):
        # first log torchmetrics if there are any
        if log_torchmetrics and len(self.metrics) > 0:
            phase_dict = {key : value for key, value in self.torchmetrics_dict.items() if phase in key}
            self.log_dict(phase_dict, on_step = False, on_epoch = True) # for these always do on_epoch

        # log all other metrics in metrics_dict
        scalar_metrics = {}
        keys = list(metrics_dict.keys()) # Iterate over a copy of the keys to avoid modification issues during iteration
        for key in keys:
            value = metrics_dict[key]
            if 'image' in key: # images
                image = self.to_pil(value)
                wandb_image = wandb.Image(image, mode="RGB")
                self.logger.experiment.log({f'{phase}_{key}': wandb_image})

            elif 'video' in key: # videos
                video_np = value.cpu().numpy()
                assert video_np.ndim != 5, "video should not include batch dimension, either fix that or add support"
                if video_np.shape[1] in [1, 3]:
                    pass  # Axes are already correct
                elif video_np.shape[-1] in [1, 3]:
                    # If video_np is (frames, height, width, channels), transpose axes
                    video_np = video_np.transpose(0, 3, 1, 2)
                else:
                    raise ValueError(f"Unexpected video shape: {video_np.shape}")
                if video_np.dtype != np.uint8:
                    video_np = (video_np * 255).astype(np.uint8)
                wandb_video = wandb.Video(video_np, fps=4, format="mp4")
                self.logger.experiment.log({f'{phase}_{key}': wandb_video})

            elif isinstance(value, torch.Tensor) and value.numel() > 1: # histogram
                self.logger.experiment.log({f"{phase}_{key}": wandb.Histogram(value.detach().cpu())})

            elif isinstance(value, torch.Tensor) and value.dim() == 0: # two types of scalar, tensor (here) and int/float (below)
                scalar_metrics[f"{phase}_{key}"] = value.detach()
            elif isinstance(value, (int, float)):
                scalar_metrics[f"{phase}_{key}"] = value
            else:
                raise ValueError(f"unsupported type/format in log_metrics, type:, {type(value)}, key: {key}")

        # Accumulate data for accurate perplexity calculation
        if self.hparams.modality == "NLP" and 'batch_total_loss' in metrics_dict and 'batch_num_tokens' in metrics_dict:
            self.step_stats[phase]["total_loss"] += metrics_dict['batch_total_loss']
            self.step_stats[phase]["total_tokens"] += metrics_dict['batch_num_tokens']
            self.step_stats[phase]["step_count"] += 1
            
            # Check if we've hit the logging interval - calculate accurate perplexity
            if hasattr(self.hparams, 'log_every_n_steps') and self.step_stats[phase]["step_count"] % self.hparams.log_every_n_steps == 0:
                self._log_accurate_perplexity_for_window(phase)

        if scalar_metrics:
            self.log_dict(scalar_metrics, sync_dist=True, prog_bar=True)
        
        if self.hparams.mcmc_step_size_learnable:
            self.log("Alpha_MCMC_Step_Size", self.model.alpha.detach())
        if self.hparams.langevin_dynamics_noise_learnable:
            self.log("Langevin dynamics step size", self.model.langevin_dynamics_noise_std.detach())
        if len(self.trainer.optimizers) == 0:
            pass # is during testing lr doesnt matter
        else:
            current_lr = self.trainer.optimizers[0].param_groups[-1]['lr'] # relies on lr for most of model being the last param
            self.log("Global_LR", current_lr)
    
    def _log_accurate_perplexity_for_window(self, phase):
        """Calculate and log accurate perplexity using token-weighted loss averaging over the last N steps."""
        stats = self.step_stats[phase]
        if stats["total_tokens"] > 0:
            # Calculate true average loss per token over the window
            avg_loss = stats["total_loss"] / stats["total_tokens"]
            # Calculate accurate perplexity
            accurate_ppl = torch.exp(torch.tensor(avg_loss, device=self.device))
            self.log(f"{phase}_accurate_perplexity", accurate_ppl)
            
            # Reset for next window
            stats["total_loss"] = 0.0
            stats["total_tokens"] = 0