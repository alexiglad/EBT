## Tools/Software being used

- The repository uses [Pytorch Lightning](https://lightning.ai/docs/pytorch/stable/common/trainer.html)
  - PL is basically a lightweight wrapper over raw pytorch that helps with:
    - Distributed training
    - Provides simple to use callbacks (see `base_model_trainer.py`)
    - Annoying cuda (`.to('cuda')`) calls
    - Speeds some things up
    - Templatizing code base, all code bases will have a similar structure. 
    - Automates the training loop, no need to call `backwards` and `zero_grad`
- Using weights and biases (WandB) as the logger
  - The wandb logger has been configured to log A TON of info including: hparams (see info tab), console logs (see logs tab, which includes model params, hparams, size of model, model layers/arch, etc), checkpoints (super helpful to not manually dig through folders!), the number of gpus, parameters/gradients, etc.
  - You can also store logs to debug in `logs/console.log` if you are not using wandb (`--no_wandb`)
- Bash and slurm scripts. Slurm scripts are basically bash scripts with an extra header on top specifying stuff about which hardware, allocation, and amount of time to use. 


## Debugging and Other Helpful Things

- [abbreviations.md](./abbreviations.md) contains, you guessed it, several abbreviations used in code throughout the repo!
  - Note that MCMC ~= optimization in the codebase, we avoid the term MCMC in the paper to avoid complexity but that's what it's commonly called in the codebase
- There is a dataloader_debugger that you can invoke with the `--debug_dataloader` flag that calls `bash job_scripts/debug/debug_dataloader.sh`.
- Unfortunately, not all hparams (or arguments) are implemented for all cases so I highly recommend using ctrl f to your advantage and potentially checking if a new hparam is implemented. This repo has evolved a lot over time.
- For debugging quickly see the argparse arguments in `train_model.py` under the DEBUGGING header
  - Several of these can be used to increase the speed of debugging as well as debug NANs/dataloader issues/fitting issues/unused parameters (detect_anomaly, debug_dataloader, overfit_batches, debug_unused_parameters)
  - For example: limit_train_batches, limit_val_batches, limit_test_batches can be used to iterate over a smaller version of the dataset
  - Several of these are supported out of the box using the [Pytorch Lightning Trainer](https://lightning.ai/docs/pytorch/stable/common/trainer.html)
- By default WandB captures the logs in the logs tab in the UI which can be useful.
- `slurm_executor.sh` is extremely useful for saving the slurm scripts used/built. If you ever find that you are launching a lot of slurm jobs and want to save the actual job script code I recommend using it.
- If CPU ram is leaking over time, chances are its because either code saving the comp graph or because of a dataloader (i.e. an HF dataloader). Check if it occurs with a dummy/synthetic dataloader, if it occurs with non multi-GPU, etc. If it is CPU RAM it is probably data; if it is GPU RAM is probably tensors.

## Known Minor Issues/TODOs

- *Important* due to perplexity being the exponential of the loss (which is a convex function), perplexities logged in wandb will always upper bound the true perplexity and are not accurate (as we log every n steps rather than waiting to average after the whole training/val epoch). Use the exponential of the logged losses in wandb for accurate ppl calculation (you should use final_loss for EBT in particular to be most accurate, as this is the loss after performing optimization).
- We have not yet implemented the KV cache for inference. Also, the EBT inference is not optimized at all (you no longer need the EBT parallelized prediction implementation, etc.) I suspect it can be made significantly faster with these fixes.
- There is no preprocessing support for anything in vision yet (i.e. pre-encoding).
- There is a small very minor bug with the learning rate of non model parameters (they will all decay to the same learning rate value of other parameters).
- Let me know of any other issues and feel free to add issues on github :)!

## Full Repository Structure

*init.py removed for your sanity :)*

```
┌── abbreviations.md
├── assets
│   └── model.png
├── base_model_trainer.py
├── CODE_INFO.md
├── data
│   ├── img
│   │   ├── coco_medium_dataset.py
│   │   ├── coco_tiny_dataset.py
│   │   ├── imagenet_dataloader.py
│   ├── nlp
│   │   ├── ai2arc_dataloader.py
│   │   ├── bigbench_dataloader.py
│   │   ├── collator.py
│   │   ├── fineweb_dataloader.py
│   │   ├── get_dataset_statistics.py
│   │   ├── gsm8k_dataloader.py
│   │   ├── lambada_dataset.py
│   │   ├── pajama_dataloader.py
│   │   ├── planbench_dataloader.py
│   │   ├── squad_dataloader.py
│   │   └── synthetic_dataset.py
│   └── vid
│       ├── aggregate_dataloader.py
│       ├── corrupt_files
│       │   ├── k400.txt
│       │   ├── ssv2.txt
│       │   └── ucf.txt
│       ├── downloader
│       │   └── ego4d.py
│       ├── kinetics_dataloader.py
│       ├── README.md
│       ├── something_dataloader.py
│       ├── ucf_dataloader.py
│       └── vid_synthetic_dataset.py
├── environment.yml
├── example_code
│   └── minimal_nlp_training_loop.py
├── gh200_requirements.txt
├── .gitignore
├── inference
│   ├── img
│   │   └── generate_image.py
│   ├── nlp
│   │   ├── eval.py
│   │   └── generate_text.py
│   └── vid
│       ├── generate_video.py
│       └── README.md
├── job_scripts
│   ├── debug
│   │   ├── debug_code2.sh
│   │   ├── debug_code.sh
│   │   └── debug_dataloader.sh
│   ├── img
│   │   ├── inference
│   │   │   ├── dit_denoising.sh
│   │   │   └── ebt_denoising.sh
│   │   ├── pretrain_denoising
│   │   │   ├── dit.sh
│   │   │   ├── ebt_s1.sh
│   │   │   └── ebt_s2.sh
│   │   └── pretrain_t2i
│   │       ├── dit.sh
│   │       ├── ebt_s1.sh
│   │       └── ebt_s2.sh
│   ├── nlp
│   │   ├── inference
│   │   │   ├── baseline_transformer.sh
│   │   │   └── ebt.sh
│   │   └── pretrain
│   │       ├── baseline_transformer.sh
│   │       ├── ebt_s1_mn.sh
│   │       ├── ebt_s1.sh
│   │       └── ebt_s2.sh
│   ├── slurm_headers
│   │   └── reference_a100.slurm
│   └── vid
│       ├── inference
│       │   ├── baseline_transformer_k400_vae.sh
│       │   ├── baseline_transformer_ssv2_vae.sh
│       │   ├── ebt_k400_vae.sh
│       │   └── ebt_ssv2_vae.sh
│       └── pretrain
│           ├── baseline_transformer_dino.sh
│           ├── baseline_transformer_vae.sh
│           ├── ebt_s1_dino.sh
│           ├── ebt_s1_vae.sh
│           └── ebt_s2_vae.sh
├── LICENSE
├── loose_requirements.txt
├── model
│   ├── ar_ebt_adaln.py
│   ├── ar_ebt_default.py
│   ├── ar_ebt_time_embed.py
│   ├── ar_transformer.py
│   ├── bi_ebt_adaln.py
│   ├── diffusion
│   │   ├── diffusion_utils.py
│   │   ├── gaussian_diffusion.py
│   │   ├── respace.py
│   │   └── timestep_sampler.py
│   ├── diffusion_transformer.py
│   ├── img
│   │   ├── dit_denoise.py
│   │   ├── dit_t2i.py
│   │   ├── ebt_denoise.py
│   │   └── ebt_t2i.py
│   ├── model_utils.py
│   ├── nlp
│   │   ├── baseline_transformer.py
│   │   ├── ebt.py
│   ├── replay_buffer.py
│   └── vid
│       ├── baseline_transformer.py
│       └── ebt.py
├── optimization.py
├── README.md
├── requirements.txt
├── slurm_executor.sh
├── structure.txt
├── train_model.py
└── utils
    ├── dataloader_debugger.py
    ├── find_corrupt_files.py
    ├── load_singularity.sh
    ├── metrics_calculator.py
    └── text_logger.py
  ```