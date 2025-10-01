# !/usr/bin/env python
# coding: utf-8
import os
from argparse import ArgumentParser
import time
import pytorch_lightning as L
from pytorch_lightning.strategies import DDPStrategy
import random
from datetime import datetime
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary
import sys
import wandb
import ast
import math
from tqdm import tqdm
import json
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.plugins.environments import SLURMEnvironment
import signal
import tempfile


# from torch.utils.tensorboard import SummaryWriter # need to implement, which involves maybe changing the forward function

from utils.dataloader_debugger import debug_dataloader
from model.model_utils import load_trained_pl_model, model_sizes, init_wandb_watch, call_style_gan_fvd
from utils import text_logger
from base_model_trainer import *
from inference.nlp.eval import nlp_eval_acc

@rank_zero_only # to ensure only one wandb run is created, if didnt do that then each GPU would create its own wandb run
def setup_wandb(args): 
    import wandb
    if wandb.run is None:
        run = wandb.init(dir="logs/", name=f'{args.run_name}', entity=f'{args.wandb_entity}', project=f'{args.wandb_project}', mode = "offline" if args.wandb_offline else "online") # this is solely used to force wandb to start tracking stdout in logs
        wandb.define_metric("__init", hidden=True)
        return run
    return None

def main(args):
    if(args.is_random_seed):
        seed_everything(random.randint(0,1000000), workers=True)
    else:
        seed_everything(33, workers=True) #33 is default
        
    if args.debug_mode:
        args.no_wandb = True
        args.detect_anomaly = True
        args.limit_train_batches = 1

    os.makedirs("./logs", exist_ok=True)

    wandb_logger = None
    if not args.no_wandb: # put this early so can capture text logs later
        run = None # need both lines since setup_wandb is a @rank_zero_only function
        run = setup_wandb(args)

        wandb_logger = WandbLogger(save_dir="logs/", name=f'{args.run_name}', entity=f'{args.wandb_entity}', project=f'{args.wandb_project}', offline = args.wandb_offline, experiment=run)
        if args.wandb_tags != None:
            wandb_logger.experiment.tags = args.wandb_tags
    else:
        console_log_file_path = os.path.join("./logs", args.console_log_filename)
        sys.stdout = text_logger.Tee(sys.stdout, console_log_file_path)
        sys.stderr = text_logger.Tee(sys.stderr, console_log_file_path)
        print("$$$$$$$$$ NOTE THAT NOT ALL STDOUT LOGS (i.e. pytorch lighting logs) ARE CAPTURED THROUGH CONSOLE LOGGER, THIS IS ONLY RECOMMENDED FOR DEBUGGING $$$$$$$$$$")

    if args.is_slurm_run:
        if not args.override_slurm_checks:
            assert args.debug_mode == False and args.detect_anomaly == False and args.limit_train_batches == 1 and args.limit_val_batches == 1 and args.limit_test_batches == 1 and args.find_unused_parameters == False and args.debug_unused_parameters == False, "for slurm run cannot have certain params set to values since am assuming are not debugging, please check values here"

        print("Current Slurm job ID:", os.environ.get('SLURM_JOBID'))
        print("Current Slurm node list:", os.environ.get('SLURM_NODELIST'))
        print("SLURM_NTASKS:", os.environ.get("SLURM_NTASKS"))
        print("SLURM_GPUS_PER_NODE:", os.environ.get("SLURM_GPUS_PER_NODE"))
        print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))

    # Set temp directory for lightning checkpointing on preemption
    temp_dir = "./logs/temp/"
    os.makedirs(temp_dir, exist_ok=True)
    tempfile.tempdir = temp_dir

    # txt_logger is no longer supported [deprecated]
    # txt_logger = text_logger.setup_custom_logger(log_filename = args.debug_log_filename, print_console = True) # this used to be set to args.print_logs but is default true now and print_logs is not used
    # args.txt_logger = txt_logger

    if args.model_size != "": # set params based off of model size
        args.num_transformer_blocks = model_sizes[args.model_size]['num_transformer_blocks']
        args.multiheaded_attention_heads = model_sizes[args.model_size]['multiheaded_attention_heads']
        args.embedding_dim = model_sizes[args.model_size]['embedding_dim']
        print("model_size", args.model_size, "args.num_transformer_blocks", args.num_transformer_blocks, "args.multiheaded_attention_heads", args.multiheaded_attention_heads, "args.embedding_dim", args.embedding_dim)

    if args.override_embedding_dim > 0:
        args.embedding_dim = args.override_embedding_dim

    if args.override_transformer_blocks > 0:
        args.num_transformer_blocks = args.override_transformer_blocks
    
    # hparam assertion sanity checks
    if args.modality == "VID":
        if args.backbone_type == "dinov2":
            assert args.embedding_dim == 0, "embedding dim defined implicitly by encoder dimensionality"
            vit_backbone_embed_dim_map = {"small": 384, "base": 768, "large": 1024, "giant": 1536} # gotten from dinov2 repo
            args.vit_backbone_dim = vit_backbone_embed_dim_map[args.vit_backbone_size]
            args.embedding_dim = args.vit_backbone_dim
        elif args.backbone_type == "vae":
            assert args.embedding_dim != 0, "must define embedding dim for vae"
        else:
            raise NotImplementedError(f"Unspported backbone type: {args.backbone_type}")
    elif args.modality == "NLP":
        assert args.embedding_dim != 0, "must define embedding dim for NLP models"
        if args.vocab_to_embed_uses_prob_dist:
            assert args.normalize_initial_condition, "if vocab_to_embed_uses_prob_dist is true must use normalize_initial_condition"
    elif args.modality == "IMG":
        args.backbone_type = "vae" # always uses vae
        assert args.embedding_dim != 0, "must define embedding dim for IMG models"
    else:
        raise ValueError(f"please add support for modality {args.modality}")
    
    # assert not(args.random_num_mcmc_steps == True and args.reconstruct_loss_only_final_step == True), "cannot have both random_num_mcmc_steps and reconstruct_loss_only_final_step set"
    # if args.ramp_up_num_mcmc_steps_every_x_epochs != -1:
    #     assert args.random_num_mcmc_steps, "random_num_mcmc_steps needs to be True"
    #NOTE should uncomment if add above hparams back
    
    args.num_nodes = int(os.getenv('SLURM_JOB_NUM_NODES', 1)) # may not exist if not using slurm so default to 1; multi node only supports slurm as of now
    print(f"SLURM_JOB_NUM_NODES: {args.num_nodes}")
    print("torch.cuda.device_count()", torch.cuda.device_count())
    if args.gpus == "-1":
        num_gpus = args.num_nodes * torch.cuda.device_count()
    elif '[' in args.gpus: # is a list
        num_gpus = len(args.gpus.split(","))
        args.gpus = json.loads(args.gpus)
    else:
        num_gpus = int(args.gpus)
    print("devices/args.gpus: ", args.gpus)

    args.total_num_workers = args.num_workers * num_gpus
    print("num_nodes", args.num_nodes, "total num_workers across all GPUs", args.total_num_workers, "num workers per GPU", args.num_workers, "num_GPUs", num_gpus)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    assert (device == torch.device('cuda') and num_gpus > 0), "using cpu instead of cuda. if you would like to proceed please remove this line and change code below to not use GPUs, otherwise check packages to ensure torch/others have cuda support"
    print(f'GPU Availability: {device}, gpus: {num_gpus}\n')
    args.num_gpus = num_gpus
    effective_batch_size = args.num_gpus * args.batch_size_per_device * args.accumulate_grad_batches
    print(f"effective_batch_size: {effective_batch_size}", "batch_size_per_device", args.batch_size_per_device)
    if args.lr_scaling_rule:
        scaled_lr = args.peak_learning_rate * effective_batch_size / 256 
        args.peak_learning_rate = scaled_lr
        print(f"Learning Rate rescaled to: {scaled_lr} based off lr_scaling_rule")
    if args.max_scheduling_steps == -1:
        args.max_scheduling_steps = args.max_steps

    model_trainer = ModelTrainer(args)

    if args.execution_mode == "finetune":
        assert args.finetuning_model_ckpt != None and args.resume_training_ckpt == "", "Must provide a checkpoint when finetuning and cannot provide a resume_training_ckpt."
        model_trainer.model = load_trained_pl_model(args.finetuning_model_ckpt, args)

    timestamp = int(time.time())
    dt_object = datetime.fromtimestamp(timestamp)
    dt_string = dt_object.strftime("%Y-%m-%d_%H-%M-%S")

    if args.debug_dataloader:
        debug_dataloader(args, model_trainer)
        return
    if args.log_model_archi:
        print(str(model_trainer.model))
        print(str(args))

    if not args.no_wandb and args.wandb_watch:
        init_wandb_watch(wandb_logger, model_trainer, args.wandb_watch_log_freq)

    if args.create_model_viz:
        # BACKLOG use tensorboard for this if active eventually. disabled for now since makes some things challenging
        return
    print(f'pytorch version: {torch.__version__}\n')

    if args.set_matmul_precision is not None: #default is highest
        torch.set_float32_matmul_precision(args.set_matmul_precision)
    
    checkpoint_filename = "epoch={epoch}-step={step}-" + args.checkpoint_monitor_string + "={"+args.checkpoint_monitor_string+":.4f}"
    checkpoint_callback = ModelCheckpoint(monitor=args.checkpoint_monitor_string, mode = args.checkpoint_monitor_mode, save_top_k=args.save_top_k_ckpts, save_last = True, dirpath=f"./logs/checkpoints/{args.run_name}_{os.environ.get('SLURM_JOBID')}_", filename=checkpoint_filename, verbose=True)
    
    for name, param in model_trainer.model.named_parameters():
        if not param.requires_grad:
            print(f"Non-trainable parameters: {name} with shape {param.shape}")
    
    if not args.only_test: #training and testing (if testing selected) as per usual
        print("$$$$$$$$$$  STARTED TRAINING  $$$$$$$$$$")
        trainer = set_trainer(args, wandb_logger, checkpoint_callback)
        resume_training_ckpt = None if args.resume_training_ckpt == "" else args.resume_training_ckpt
        trainer.fit(model_trainer, ckpt_path=resume_training_ckpt)
        
        if args.run_testing_after_training:
            args.only_test_model_ckpt = checkpoint_callback.best_model_path
            best_model = ModelTrainer.load_from_checkpoint(
                checkpoint_callback.best_model_path, 
                hparams=args
            )
            clear_cache()
            print(f"best model path that will be used during testing {checkpoint_callback.best_model_path}")
            print("$$$$$$$$$$  STARTED TESTING AFTER TRAINING  $$$$$$$$$$")
            raise NotImplementedError("need to test this with newer PL")
            # test_trainer = L.Trainer(logger=wandb_logger,devices=1,num_nodes=1) NOTE tested this code does not work gets stuck, see thread, TODO test with newer PL
            #warning reference - https://github.com/Lightning-AI/lightning/issues/12862
            best_model.eval()
            trainer.test(best_model)
        clear_cache()
    else: #only testing
        print("$$$$$$$$$$  ONLY TESTING MODEL ([NO]) TRAINING  $$$$$$$$$$")
        assert args.only_test_model_ckpt != None, "Must supply pretrained model when only testing"
        run_name, log_file_name = args.only_test_model_ckpt.split("/")[-2:]
        args.save_generation_logs_dir = os.path.join(args.infer_output_dir, args.modality, args.dataset_name, run_name, log_file_name.replace(".ckpt", ""))
        if os.path.exists(args.save_generation_logs_dir): # remove any existing logs
            shutil.rmtree(args.save_generation_logs_dir)

        
        checkpoint = torch.load(args.only_test_model_ckpt, weights_only=False)
        pretrained_hparams = checkpoint['hyper_parameters']

        default_args = vars(args).copy() # NOTE this is so can test older models trained with older code as well as use newer hparams (for inference) on pretrained models, may be finicky feel free to tweak
        for key, value in default_args.items():
            if key not in pretrained_hparams:
                pretrained_hparams[key] = value
                print(f"MISSING PARAMETER IN PRETRAINED CHECKPOINT: Using args set value for missing parameter in pretrained checkpoint '{key}': {value}")
            if key.startswith("infer_"):
                pretrained_hparams[key] = value
                print(f"OVERRIDING PARAMETER IN PRETRAINED CHECKPOINT FOR INFERENCE: Using args set value for inference parameter '{key}': {value}")

        if args.modality == "VID":
            pretrained_hparams["modality"] = "VID" # this is just to test old models with a refactor of CV -> VID

        model = ModelTrainer(pretrained_hparams)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        model_trainer = ModelTrainer(args, trained_model=model.model) # need to use args as we have to use the model most recently passed in for inference

        trainer = set_trainer(args, wandb_logger, checkpoint_callback, stage = "test")
        model_trainer.model.eval()
        trainer.test(model_trainer)
        if args.modality == "NLP": # can have modality specific logic here for inference
            if args.execution_mode == "inference":
                em_score, f1_score = nlp_eval_acc(os.path.join(args.save_generation_logs_dir, "results.jsonl"))
                trainer.logger.experiment.log({"em_score": em_score, "f1_score": f1_score})
        elif args.modality == "VID":
            if args.infer_generate_video:
                print("calling style gan FVD code on generated video dataset, NOTE THIS CODE MAY NOT WORK AS EXPECTED or get stuck")
                fvd, fid = call_style_gan_fvd(args)
                trainer.logger.experiment.log({"test_fvd": fvd, "test_fid": fid})
        elif args.modality == "IMG":
            pass # no post test code for denoising, for t2i could call FID code here if desired
        else:
            raise NotImplementedError(f"no post test evaluation setup for this modality: {args.modality} yet")

def set_trainer(args, wandb_logger, checkpoint_callback, stage = "train"):
    torch.autograd.set_detect_anomaly(args.detect_anomaly) #NOTE seems pl detect anomaly is not working so manually set it here

    if args.find_unused_parameters: #for if ever need more manipulation
            args.distributed_strategy = DDPStrategy(find_unused_parameters = True)
        # else:
        #     pass
        # if having issues with strategy try 'ddp_spawn' instead of 'ddp'
    args.overfit_batches = int(args.overfit_batches) if int(args.overfit_batches) == args.overfit_batches else args.overfit_batches
    profiler = None if args.profiler == "" else args.profiler
    gradient_clip_val = args.gradient_clip_val if args.gradient_clip_val > 0 else None
    limit_val_batches = 0 if args.overfit_batches > 0 else args.limit_val_batches
    val_check_interval = args.val_check_interval if args.val_check_interval == 1.0 else args.val_check_interval * args.accumulate_grad_batches  #NOTE the reason we mult by args.accumulate_grad_batches is because of this bug https://github.com/Lightning-AI/pytorch-lightning/issues/12205
    limit_test_batches = args.limit_test_batches if args.limit_test_batches == 1 else args.limit_test_batches * args.accumulate_grad_batches
    trainer = L.Trainer(
        default_root_dir=f"./logs/checkpoints/{args.run_name}_{os.environ.get('SLURM_JOBID')}_/preempts",
        accelerator="auto",
        devices = args.gpus,
        num_nodes=args.num_nodes,
        precision=args.float_precision,
        max_steps=args.max_steps,
        logger=wandb_logger,
        enable_model_summary=args.log_model_archi,
        callbacks = [checkpoint_callback, ModelSummary(max_depth=-1)],
        strategy = args.distributed_strategy, 
        plugins=[SLURMEnvironment(requeue_signal=signal.SIGTERM)],
        enable_checkpointing=True,
        fast_dev_run = args.fast_dev_run,
        num_sanity_val_steps = args.val_sanity,
        limit_train_batches = args.limit_train_batches,
        limit_val_batches = limit_val_batches,
        limit_test_batches = limit_test_batches,
        detect_anomaly=args.detect_anomaly,
        gradient_clip_val=gradient_clip_val,
        overfit_batches=args.overfit_batches,
        profiler=profiler,
        val_check_interval=val_check_interval,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        deterministic=args.deterministic,
        log_every_n_steps=args.log_every_n_steps,
        accumulate_grad_batches=args.accumulate_grad_batches,
        inference_mode=False # set inference mode to false to get grad for models like ebt during testing
    )
    return trainer
    

def clear_cache():
    torch.cuda.empty_cache()
    

if __name__ == '__main__':
    parser = ArgumentParser()

    #SUPER IMPORTANT HPARAMS--always set these properly #############################################

    parser.add_argument("--run_name", help="run name, should include model name in it for important runs (see slurm scripts)", default="test")

    parser.add_argument("--modality", help="is the model being trained for NLP, VID, IMG. default is VID, and most of the code has been designed around VID.", type=str, default="VID")

    parser.add_argument("--model_name", help="model_type/name", default='ebt')

    parser.add_argument("--model_size", help="model size, is used to set num_transformer_blocks, multiheaded_attention_heads, and embedding_dim", choices=model_sizes.keys(), default='xxs')

    #NOTE not all below hparams are implemented for all models, so please check before using one     

    # VID SPECIFIC PARAMS ############################################################################################################

    parser.add_argument("--vit_backbone_size", help="small, base, large or giant", type=str, default="base")

    parser.add_argument("--backbone_type", help="backbone type for VID encoder", choices=["dinov2", "vae"], type=str, default="dinov2")

    parser.add_argument("--time_between_frames", help="time (seconds) between frames of video", type=float, default=1.0)
    
    parser.add_argument("--sampling_rate", help="how many frames to draw a single video frame from. e.g. sampling rate=4 will have three frames in between each sample. overrides time_between_frames", type=float, default=0.0)
    
    parser.add_argument("--use_raw_framerate", help="whether to use dataset default framerate with a sampling rate of 1. this overrides the time_between_frames and sampling_rate parameters.", action="store_true", default=False)

    parser.add_argument("--vae_normalization", help="whether to use the commonly used vae normalization of [-1, 1].", action="store_true", default=False)

    parser.add_argument("--sdxl_vae_standardization", help="whether to standardize vae encodings according to sdxl by multiplying by 0.18215.", action="store_true", default=False)

    parser.add_argument("--weight_tie_vae_proj", help="for baseline transformer, weight ties the projection to vae latent space", action="store_true", default = False)
     
    # NLP SPECIFIC PARAMS ############################################################################################################

    parser.add_argument("--tokenizer", help="tokenizer for nlp tasks", type=str, default="EleutherAI/gpt-neox-20b")
    
    parser.add_argument("--pretokenize_dataset", help="whether to pretokenize the dataset and save that or just tokenize as loading the dataset. tokenizing the dataset takes a long time and may need to be done using debug dataloader. due to a bug with HF does not always work reliably, may get stuck. not currently implemented for fine-tuning datasets", action="store_true", default=False)

    parser.add_argument("--normalize_initial_condition", help="makes initial condition a normalized probability distribution if modality is NLP, helps a ton with stability", action="store_true", default = False)

    parser.add_argument("--normalize_initial_condition_only_first_step", help="makes initial condition a normalized probability distribution if modality is NLP ONLY on first MCMC step, by default does for all steps. this didnt work very well", action="store_true", default = False)

    parser.add_argument("--sharpen_predicted_distribution", help="scales the temperature of the distribution, higher temp means less sharp lower temp sharpens distribution. divides by this number. didnt work very well", type=float, default=0.0)

    parser.add_argument("--vocab_to_embed_uses_prob_dist", help="makes initial condition a normalized probability distribution if modality is NLP, this worked decently well and may be more data efficient", action="store_true", default = False)

    parser.add_argument("--soften_target_prob_dist", help="softens the target prob dist differently across different mcmc steps, making it harder during later MCMC steps. didnt work very well", type=float, default=0.0)

    # IMG SPECIFIC PARAMS ########################################################################################################################

    parser.add_argument("--image_task", help="task for image modality, default is text to image (t2i)", choices=["t2i", "denoising"], type=str, default="t2i")

    parser.add_argument("--denoise_images_noise_level", help="when denoising is the image task what std of gaussian noise added should be", type=float, default=1)

    parser.add_argument("--randomize_denoise_images_noise_level", help="(for non diffusion models) denoise the original image to a random extent, rather than just the max", action="store_true", default = False)

    parser.add_argument("--clip_text_encoder_size", help="size for clip text encoder", choices=["small", "base", "large", "xl"], type=str, default="base")

    parser.add_argument("--patch_size", help="patch size from DiT paper, larger means less compute", choices=[2, 4, 8, 16], type=int, default=8)

    parser.add_argument("--log_image_every_n_steps", help="logs a generated image every n training steps", type=int, default=500)

    #EBT MCMC ##################################################################
    #NOTE in the code we broadly refer to MCMC/optimization interchangeably, although in the paper we refrain from the term mcmc

    parser.add_argument("--langevin_dynamics_noise", help="is the standard deviation of noise to use that is mean centered", type=float, default=0.0)
    
    parser.add_argument("--langevin_dynamics_noise_learnable", help="is the std of noise for langevin dynamics learnable", action="store_true", default = False)

    parser.add_argument("--no_langevin_during_eval", help="dont use langevin dynamics during validation", action="store_true", default = False)
    
    parser.add_argument("--mcmc_step_size", help="is size of optimization step, or alpha in the paper, kinda like LR, can be learned param", type=float, default=60.0)
    
    parser.add_argument("--mcmc_step_size_learnable", help="makes mcmc_step_size a learnable param", action="store_true", default = False)
    
    parser.add_argument("--mcmc_step_size_lr_multiplier", help="learning rate multiplier for mcmc step size, so to get lr of mcmc step size take lr multiply by this value", type=float, default=5000.0)

    parser.add_argument("--randomize_mcmc_step_size_scale", help="randomize the value of mcmc_step_size by a factor specified, i.e. if is 2 will mult by 2 and div by 2 and thats the range to sample from uniformly", type=float, default=1)
    
    parser.add_argument("--mcmc_num_steps", help="number of MCMC steps, try 2-5, check data samples as well to see how many we need. NOTE if are using time embed or adaln is the number of energy landscapes", type=int, default=2)

    parser.add_argument("--randomize_mcmc_num_steps", help="makes mcmc_num_steps random, each step at each landscape is repeated uniform(1, 1+randomize_mcmc_num_steps) times (unless randomize_mcmc_num_steps_min is set, then thats the min value). if ebt_type is default each landscape is the same, so it effectively just randomized mcmc_num_steps", type=int, default=0)

    parser.add_argument("--randomize_mcmc_num_steps_final_landscape", help="makes it so the randomize_mcmc_num_steps param only applies to the final energy landscape when using non default EBT", action="store_true", default = False)

    parser.add_argument("--randomize_mcmc_num_steps_min", help="makes it so there is a minimum number of randomized mcmc steps", type=int, default=0)
    
    parser.add_argument("--denoising_initial_condition", help="['random_noise', 'most_recent_embedding', or 'zeros'], what condition to start off for denoising. random_noise is most likely the best and what is used for every experiment in the paper, most_recent_embedding may be helpful for VID but probably not for NLP, or can condition on zeros (vector)", type=str, default="random_noise")

    parser.add_argument("--gaussian_random_noise_scaling", help="scales the std of the gaussian sampled from for random noise denoising_initial_condition. is mainly for NLP ebt model (not impl in other modalities for now)", type=float, default=1)

    parser.add_argument("--clamp_futures_grad", help="clamps grad of predictions (used to be called futures when was doing AR). not used anymore but can help with stability", action="store_true", default=False)

    parser.add_argument("--clamp_futures_grad_max_change", help="max total change during mcmc to clamp to (possibly divides by num mcmc steps), used by above. not used anymore", type=float, default=9.0)

    parser.add_argument("--absolute_clamp", help="clamps the absolute value of predicted_tokens to be within range [-val, val]. not used anymore", type=float, default=0.0)

    parser.add_argument("--clamp_max_after_warm_up", help="clamps the absolute value of predicted_tokens to be within range [-val, val], after warming up. not used anymore", type=float, default=0.0)

    parser.add_argument("--ebt_type", help="type of energy based transformer to use, inspired by DiT paper.", choices=["default", "time_embed", "adaln", "adaln_zero"], type=str, default="default")

    parser.add_argument("--ebt_norm", help="type of norm to use for energy based transformer, NOTE is only supported for ebt_time_embed. not used anymore didnt work better than default rms from llama2", choices=["rms", "none", "layer", "ebm_backwards_norm", "dyt"], type=str, default="rms")

    parser.add_argument("--dyt_alpha_init", help="initial value for alpha in dyt layer norm, from paper https://jiachenzhu.github.io/DyT/. didnt work well increased instability", type=float, default=0.5)

    parser.add_argument("--ebt_act_func", help="activation function to use for energy based transformer, NOTE is only supported for ebt_time_embed. silu (default from llama2) worked best", type=str, default="silu")

    parser.add_argument("--truncate_mcmc", help="truncate mcmc and only use final step of loss to calculate, for S2 models", action="store_true", default=False)

    parser.add_argument("--mcmc_replay_buffer", help="enables a replay buffer for MCMC, particularly S2 models", action="store_true", default=False)

    parser.add_argument("--mcmc_replay_buffer_sample_bs_percent", help="number of samples to retrieve in replay buffer, as a percentage of batch size per device", type=float, default=0.5)

    parser.add_argument("--mcmc_replay_buffer_size", help="number of total samples to store in replay buffer", type=int, default=192) # for 40 GB A100 192 is the max

    parser.add_argument("--no_mcmc_detach", help="dont detach between mcmc steps, probably need to use for S2 models but can increase instability due to longer gradient computation graphs", action="store_true", default=False)

    parser.add_argument("--contrastive_loss", help="uses a contrastive loss to shape the landscape of EBM, idea from IRED paper https://arxiv.org/abs/2406.11179", action="store_true", default=False)

    parser.add_argument("--contrastive_loss_coeff", help="coefficient for contrastive loss, didnt work well not used", type=float, default=0.0005)

    parser.add_argument("--discrete_contrastive_loss_true_logit_val", help="value for contrastive loss discrete logits, didnt work well not used", type=float, default=0.0)

    parser.add_argument("--learnable_process_memory", help="allows for a learnable process memory, which will be mapped to the same embed as the vocab dist using an MLP. didnt explore enough but worth trying out :) feel free to reach out to discuss", action="store_true", default=False)

    parser.add_argument("--process_memory_type", help="type of learnable process memory", choices=['add', 'gate', 'residual_gate'], type=str, default=None)
    
    parser.add_argument("--process_memory_linear_layer", help="allows for the process memory to have an extra linear layer to process it", action="store_true", default=False)

    # removed hparams that could be useful to try again (if enable add impl (ask alexi) and uncomment above lines with NOTE that use these params)

    # parser.add_argument("--reconstruct_loss_only_final_step", help="this will make it so the reconstruction loss will only be based on the final step. this is useful to allow for more MCMC to occur without lowering bs.", action="store_true", default = False)

    # parser.add_argument("--random_num_mcmc_steps", help="makes it so out of the given num or mcmc steps, has random chance to do 1-N (where N is num provided). Only calcs rec loss on last 2 steps.", action="store_true", default = False)
    
    # parser.add_argument("--ramp_up_num_mcmc_steps_every_x_epochs", help="every X epochs ramps up num mcmc steps, should use with random_num_mcmc_steps", type=int, default=-1) # shuld prob replace epochs for steps to be more modality agnostic

    #EBT [mostly deprecated] THINGS from a past life ################################################################
        
    parser.add_argument("--reconstruction_coeff", help="coefficient for reconstruction loss, was for when doing multi objective but not just is always 1", type=float, default=1.0)
    
    parser.add_argument("--out_of_bounds_loss_coeff", help="coefficient for oob loss for mcmc, was for when was trying to regularize energy landscape, didnt work well not used", type=float, default=0.0)

    # all below are for energy loss (which is no longer used, just rec loss is used)

    parser.add_argument("--energy_loss_coeff", help="coefficient for energy prediction loss, was for when was trying to regularize energy landscape, didnt work well not used", type=float, default=0.0)
    
    parser.add_argument("--energy_loss_hinge", help="hinge loss margin, so only predicts energy being close up to this amount, helps with inherent randomness in cosine sim", type=float, default=0.0)
    
    parser.add_argument("--energy_loss_fn", help="l1_loss, MSE, smooth_l1_loss", type=str, default="MSE")
        
    parser.add_argument("--embeddings_distance_fn", help="for comparing embeddings of predictions (energy loss coeff) - euclidean, normalized_euclidean, cosine, manhattan", type=str, default="cosine")
    
    parser.add_argument("--scale_cosine_sim_decay", help="scale cosine sim so has more variation < 0.5 and less above 0.5, try 0 (for none) 2, 4, 6, 8 for energy_loss_coeff", type = int, default=7)

    # DIFFUSION ################################################################
    parser.add_argument("--diffusion_steps", help="number of steps in diffusion", type = int, default=1000)

    parser.add_argument("--use_deterministic_reverse", help="makes it so the reverse process is deterministic for generation and uses ddim instead of ddpm with learned sigma", action="store_true", default = False)

    parser.add_argument("--infer_increase_steps", help="inference number of steps to increase to", type = int, default=0)

    parser.add_argument("--infer_recurse_diffusion_n_times", help="inference number of times to recurse diffusion on its own outputs, helpful for denoising OOD data. did work better than increase steps", type = int, default=0)

    #MODEL AND ARCHITECTURE ##################################################################

    # transformer specific ############################################

    parser.add_argument("--context_length", help="context length for AR models, i.e. for language model (commonly 256) or video model (commonly 16)", type=int, default=0)
          
    parser.add_argument("--num_transformer_blocks", help="number of transformer blocks, uses default from model size specified", type=int, default=12)
    
    parser.add_argument("--multiheaded_attention_heads", help="number of attention heads for transformer, uses default from model size specified", type=int, default=2)

    ######################################################################

    parser.add_argument("--embedding_dim", help="embedding dimension for transformers, if are using a model size this is automatically set", type=int, default=384)

    parser.add_argument("--ffn_dim_multiplier", help="how much wider than the embedding dim the transformer FFN dim should be", type=float, default=None)

    parser.add_argument("--override_embedding_dim", help="override the embedding dimension for a transformer. do not use unless you know what you are doing since it will likely cause issues and mess up scaling trends", type=int, default=0)

    parser.add_argument("--override_transformer_blocks", help="override the number of blocks for a transformer. do not use unless you know what you are doing since it will likely cause issues and mess up scaling trends", type=int, default=0)

    parser.add_argument("--num_modality_processing_mlp_layers", help="number of linear layers in the MLP for processing each modality, need at least 1", type=int, default=1) # NOTE not currently used or implemented

    parser.add_argument("--weight_initialization_method", help="xavier or he", type=str, default="xavier")

    parser.add_argument("--weight_initialization_gain", help="gain of the weight init, see https://pytorch.org/docs/stable/nn.init.html, default is equiavalent to linear gain which is 1. tried tweaking this for EBT didnt help", type=float, default=1.0)

    # seperate MLP specific #####################################################################
    # NOTE, these are not for transformer MLP are for seperate MLPs
    parser.add_argument("--mlp_dropout", help="dropout in basic MLP", type=float, default=0.0)
                
    parser.add_argument("--mlp_layer_norm", help="layer normalization", action="store_true", default=False)

    parser.add_argument("--mlp_dim_multiplier", help="how much wider than the embedding dims the final layer projector should be", type=float, default=2.0)

    parser.add_argument("--mlp_hidden_layers", help="number of linear layers in MLP after encoder, if is 1 is just a linear layer", type=int, default=0)

    ################
    
    parser.add_argument("--encoder_lr_divider", help="how much to divide lr for encoder", type=float, default=100.0)  

    #HARDWARE###################################################################

    parser.add_argument("--gpus", help="number of gpus or gpus list, -1 uses all GPUs. use -1 for multinode, if want to specify which GPUs to use specify as comma seperated str with brackets e.g. [0, 1]", default="-1")
    
    parser.add_argument("--distributed_strategy", help="distributed strategy - ddp_spawn, ddp, fsdp_native, or None", default='ddp')


    #TRAINING#########################################################

    parser.add_argument("--peak_learning_rate", help="peak learning rate after warm up", type=float, default=0.02)

    parser.add_argument("--batch_size_per_device", help="batch size (PER DEVICE!!!, not effective). to get effective_batch_size is num_gpus * batch_size_per_device * accumulate_grad_batches, see above)", type=int, default=2)

    parser.add_argument("--accumulate_grad_batches", help="number of batches to accumulate before stepping optimizer, 1 is regular", type=int, default=1)

    parser.add_argument("--gradient_clip_val", help="maximum value for gradient clipping", type=float, default=1.0)
    
    parser.add_argument("--is_random_seed", help="is_random_seed", action="store_true", default=False)
    
    parser.add_argument("--deterministic", help="ensures results are determnistic, may run a bit slower, sets flag on pl trainer and sets workers for dataloader", action="store_true", default=False)

    parser.add_argument("--execution_mode", type=str, choices=["pretrain", "finetune", "inference"], default="pretrain")
    
    parser.add_argument("--finetuning_model_ckpt", help="model ckpt when finetuning", type=str, default=None)

    #METRICS#########################################################
    #NOTE reported metrics in wandb will be averages over the time span they are being computed, so the final one in an epoch will be the most accurate

    parser.add_argument("--metrics_list", help = "list of metrics to use", nargs='+', default = [])
    
    parser.add_argument("--metrics_average_type", help="type of average to use for accuracy - macro, micro, weighted - see torchmetrics docs for more info", type=str, default="micro")

    parser.add_argument("--num_classes", help="the number of classes, must set if using metrics, see torchmetrics docs for more info", type=int, default=-1)

    parser.add_argument("--metrics_task", help="the type of task being done ['binary', 'multiclass' or 'multilabel'], see torchmetrics docs for more info, must set if using metrics_list", type=str, default="")
    
    #OPTIMIZER AND LR SCHEDULER#########################################################
    
    parser.add_argument("--weight_decay", help="weight decay to use", type=float, default=0.01)
    
    parser.add_argument("--beta1", help="exponential decay rate for first moment estimate", type=float, default=0.9)
    
    parser.add_argument("--beta2", help="exponential decay rate for second movement estimate", type=float, default=0.999)

    parser.add_argument("--lr_scaling_rule", help="the LR will be scaled according to the rule LR = base_lr * effective_batch_size / 256. is useful for prototyping and is popular in vision SSL. effective_batch_size is based off bs * num_gpus * accumulate_grad_batches", action="store_true", default=False)

    parser.add_argument("--min_lr_scale", help="the most the lr will be scaled down during cosine decay", type=int, default=10)

    parser.add_argument("--max_steps", help="max number of steps for training", type=int, default=1000000)

    parser.add_argument("--max_scheduling_steps", help="max number of steps used for lr/other hparam scheduling. in general should be the same as max_steps but can be different if dont want to do a full run but want the lr and other things to be scheduled the same. similar to V jepa paper. if is not set will default to max_steps value", type=int, default=-1)
        
    parser.add_argument("--warm_up_steps", help="number of steps to increase the LR linearly over before hitting specified peak LR, starts from 0 unless warm_up_base_lr_divider is set and does linear warm up", type=int, default=10000)
    
    parser.add_argument("--warm_up_base_lr_divider", help="lr divider for when doing linear learning rate warm up. if is set to -1 then does warm up from 0", type=float, default=-1)
    
    parser.add_argument("--optimizer", help="used to turn on different optimizers. current options include adamw (default), lars, stableadamw", type=str, default="adamw")
    
    parser.add_argument("--lars_trust_coeff", help="exponential decay rate for second movement estimate", type=float, default=0.001)
    
    parser.add_argument("--lars_exclude_bias_bn_wd", help="excludes bias and batch norm from Lars adaptation and weight decay", action="store_true", default=False)

    #DATASET AND DATALOADER #########################################################

    parser.add_argument("--num_workers", help="num_workers per GPU. idea to do per GPU gotten from https://discuss.pytorch.org/t/guidelines-for-assigning-num-workers-to-dataloader/813/", type=int, default=4)

    parser.add_argument("--prefetch_factor", help="prefetch factor for dataloader", type=int, default=None)
    
    parser.add_argument("--dataset_name", help="dataset name", default="ucf101")
    
    parser.add_argument("--dataset_dir", help="dataset base directory", default="")
    
    parser.add_argument("--image_dims", help="List of image dimensions", nargs='+', type=int, default = [224, 224])
    
    parser.add_argument("--no_randomness_dataloader", help="makes dataloader have no randomness by only sampling from start", action="store_true", default=False)
        
    parser.add_argument("--preprocess_data", help="center crops images - helps to avoid black bars on sides which cause issues with unstable gradient. not used anymore", action="store_true", default=False)
    
    parser.add_argument("--crop_all_samples", help="if preprocess data is enabled this will crop all samples. not used anymore", action="store_true", default=False)
    
    parser.add_argument("--custom_image_normalization", help="whether to normalize data according to each dataset's std and mean", action="store_true", default=False)

    parser.add_argument("--use_rand_caption", help="whether to use a random caption for image generation rather than always the first caption", action="store_true", default=False)

    parser.add_argument("--ffprobe_path", help="path to ffprobe binary", default="")

    parser.add_argument("--validation_split_pct", help="if training and validation set are being split - which percent is validation", type=float, default=0.1)
    
    parser.add_argument("--test_split_pct", help="if training, validation, and test set are being split - which percent is test. by default this is not used", type=float, default=0)
    
    parser.add_argument("--limit_train_batches", help="percent of training dataset to use, or if > 1 is num batches", type=float, default=1.0)
    
    parser.add_argument("--limit_val_batches", help="percent of validation dataset to use, or if > 1 is num batches", type=float, default=1.0)
    
    parser.add_argument("--limit_test_batches", help="percent of testing dataset to use, or if > 1 is num batches", type=float, default=1.0)
    
    parser.add_argument("--val_sanity", help="number of sanity validation steps", type=int, default=0)
    
    parser.add_argument("--val_check_interval", help="interval to do validation per training epoch, 1 means once per epoch. useful if epochs are too large to wait to do validation", type=float, default=1.0)

    parser.add_argument("--check_val_every_n_epoch", help="how many epochs to wait before doing validation. is useful when you have a small dataset and dont want to do validation every time", type=int, default=1)

    #TESTING / INFERENCE#########################################################

    parser.add_argument("--run_testing_after_training", help="evaluate on test dataset. by default is off since repo is mostly pre-training, and may not work :/", action="store_true", default=False)

    parser.add_argument("--only_test", help="just testing no training, used passed in model checkpoint", action="store_true", default=False)

    parser.add_argument("--only_test_model_ckpt", help="model ckpt when only testing", type=str, default=None)
    
    parser.add_argument("--infer_max_gen_len", help="[Inference] Maximum number of tokens/frames/sequence to generate", type=int, default=256)

    parser.add_argument("--infer_echo", help="[Inference] Include input prompt in the output", type=bool, default=False)
    
    parser.add_argument("--infer_output_dir", type=str, default="./logs/inference")

    # NLP INFERENCE ########################################################################
    
    parser.add_argument("--infer_temp", help="[Inference] Sampling temperature (higher = more random)", type=float, default=0.6)
    
    parser.add_argument("--infer_topp", help="[Inference] Nucleus sampling probability threshold", type=float, default=0.9)
    
    parser.add_argument("--infer_topk", help="[Inference] Limit sampling to top k most likely tokens", type=int, default=None)
    
    parser.add_argument("--infer_logprobs", help="[Inference] Return log probabilities of generated tokens", type=bool, default=False)

    # VID INFERENCE ########################################################################

    parser.add_argument("--infer_video_condition_frames", help="[Inference] number of frames to condition generation on", type=int, default=3)

    parser.add_argument("--infer_generate_video", help="[Inference] whether to get FVD and FID metrics", action="store_true", default=False)

    # EBT SPECIFIC INFERENCE/TESTING PARAMS FOR ADVANCED INFERENCE ##################################################

    parser.add_argument("--infer_ebt_advanced", help="[Inference] Does advanced inference for EBT, including hparams for more mcmc steps, override alpha, generating many samples and choosing the best, langevin dynamics, etc", action="store_true", default=False)

    parser.add_argument("--infer_ebt_num_steps", help="[EBT Inference] Number of MCMC steps to take, if is time embed/adaln will do this number of steps for each mcmc discrete time step", type=int, default=1)

    parser.add_argument("--infer_ebt_override_alpha", help="[EBT Inference] Step size for MCMC, default is 0, if is nonzero will override the trained value", type=float, default=0)

    parser.add_argument("--infer_generated_samples", help="[EBT Inference] Number of generated samples to create per sample, and then choose using infer_energy_sampling_technique", type=int, default=1)

    parser.add_argument("--infer_debug_sample_distances", help="[Inference] Print the distances between samples when generating many samples when doing infer_generated_samples > 1", action="store_true", default=False)

    parser.add_argument("--infer_plot_energy_landscape", help="[Inference] Plot the energy landscape and the corresponding solution distances. didnt work very well unsure if is due to bug", action="store_true", default=False)
    
    parser.add_argument("--infer_langevin_dynamics_noise", help="[EBT Inference] Langevin dynamics noise during inference, default is 0", type=float, default=0)

    parser.add_argument("--infer_langevin_first_step", help="[Inference] Makes it so the langevin dynamics only applies to the first step", action="store_true", default=False)

    parser.add_argument("--infer_energy_sampling_technique", help="the way to sample from EBM, default is min which means choose the sample with the lowest energy, other options are max_gap which chooses the sample with highest gap from starting energy and lowest energy", choices = ["min", "max_gap", "max"], type=str, default="min")

    parser.add_argument("--infer_accept_lower_energies", help="[Inference] Makes it so only accept a step if it has lower energy", action="store_true", default=False)

    parser.add_argument("--infer_steps_final_landscape", help="[Inference] Makes it so infer_ebt_num_steps only affects the final energy landscape in the case of EBT with time_embed/adaln", action="store_true", default=False)

    parser.add_argument("--infer_alpha_final_landscape", help="[Inference] Makes it so infer_ebt_override_alpha only affects the final energy landscape in the case of EBT with time_embed/adaln", action="store_true", default=False)

    #WANDB##################################################################

    parser.add_argument("--no_wandb", help="no wandb", action="store_true", default=False)

    parser.add_argument("--wandb_entity", help="wandb entity", default='')

    parser.add_argument("--wandb_project", help="wandb project name", default='EBT')

    parser.add_argument("--wandb_tags", help="wandb tags to add", nargs='+', default=None)

    parser.add_argument("--wandb_offline", help="set wandb to offline mode", action="store_true", default=False)

    parser.add_argument("--wandb_watch", help="turns on watch mode for wandb - expensive so only use for debugging", action="store_true", default=False) 

    parser.add_argument("--wandb_watch_log_freq", help="number of steps to log for wandb watch. is higher since is a bit expensive", type = int, default=1000)  

    #LOGGING##################################################################

    parser.add_argument("--debug_log_filename", help="filename of log, deprecated", default='debug.log')

    parser.add_argument("--console_log_filename", help="filename of log, used when no_wandb is active", default='console.log')

    parser.add_argument("--print_logs", help="[deprecated] print to console, default false", action="store_true", default=False)

    parser.add_argument("--log_model_archi", help="log model architecture", action="store_true", default=False)

    parser.add_argument("--log_gradients", help="logs gradients at every step to wandb to debug them", action="store_true", default=False)

    parser.add_argument("--log_every_n_steps", help="turns on logger freq via pl, not advised to use this", type=int, default=50)

    # CHECKPOINTING ##################################################################

    parser.add_argument("--resume_training_ckpt", help="checkpoint to resume training from, use absolute",type=str, default="")     

    parser.add_argument("--checkpoint_monitor_string", help="string to use to monitor for saving checkpoint. supported by PL callback", type=str, default="valid_loss")

    parser.add_argument("--checkpoint_monitor_mode", help="monitoring mode for checkpoint_monitor_string, either ['min', 'max']. if is loss do min, if is a metric like accuracy do max", type=str, default="min")

    parser.add_argument("--save_top_k_ckpts", help="number of ckpts to save when doing val (saves the ones with best metrics using checkpoint monitor string and mode defined). -1 means save all", type=int, default=10)

    #PRECISION#########################################################################

    parser.add_argument("--set_matmul_precision", help="set math mult precision - \"medium\", \"high\", or \"highest\" ", default=None)

    parser.add_argument("--float_precision", help="float precision, pl recommends 16-mixed/bf16-mixed, also has by default 32-true", type=str, default="32-true")

    # SPEED ##################################################################

    parser.add_argument("--compile_model", help="compiles the model using torch.compile", action="store_true", default=False)

    #SLURM#########################################################################

    parser.add_argument("--is_slurm_run", help="please set to true if doing slurm run, as of now just stops capturing console logs", action="store_true", default=False)
    
    parser.add_argument("--override_slurm_checks", help="dont use slurm checks to assert that certain conditions are true (i.e. train/test limit_batches is default value)", action="store_true", default=False)
    
    #DEBUGGING#######################################################################

    parser.add_argument("--debug_mode", help="turns debug mode on where dataset returned is very small, no_wandb is on and detect anomaly is on", action="store_true", default=False)

    parser.add_argument("--fast_dev_run", help="turns fast_dev_run for trainer on, makes it just do one training epoch and one val epoch", action="store_true", default=False)

    parser.add_argument("--debug_dataloader", help="makes mode to just debug dataloader", action="store_true", default=False)

    parser.add_argument("--overfit_batches", help="if nonzero will overfit to specified num/percent of batches", type=float, default=0.0)

    parser.add_argument("--profiler", choices=["simple", "advanced"], type=str, default="")

    parser.add_argument("--no_shuffle", help="stops shuffling - helpful for debugging", action="store_true", default=False)

    parser.add_argument("--create_model_viz", help="creates model for visualization", action="store_true", default=False)

    parser.add_argument("--detect_anomaly", help="turns on anomaly detection mode", action="store_true", default=False)

    parser.add_argument("--find_unused_parameters", help="turns on pl find unused params mode - DO NOT KEEP ON for actual training, helpful if want to debug. this uses DDPStrategy and ignores distributed_strategy", action="store_true", default=False)

    parser.add_argument("--debug_unused_parameters", help="makes it so it tracks which params are used to find the params that are causing the unused params issue. need to do some things in base_model_trainer so ctrl f this hparam to see the NOTEs", action="store_true", default=False)

    parser.add_argument("--manual_gc_collect_every_n_steps", help="manually call gc collect every n steps, can be done to prevent CPU RAM memory 'leak'", type = int, default=-1)

    parser.add_argument("--debug_videos", help="debug generated videos in a grid", action="store_true", default=False)

    parser.add_argument("--preencode_video", help="boolean to toggle video dataset preencoding", action="store_true", default=False)

    parser.add_argument("--hdf5_file_path", help="path to preencoded video dataset hdf5 file", type=str, default="")

    parser.add_argument("--vid_preencoding_ratio", help="ratio of videos to preencode", type=float, default=1.0)


    ########################################################################

    args = parser.parse_args()
    main(args)