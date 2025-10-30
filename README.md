# ⚡️🧠🤖 Energy-Based Transformers are Scalable Learners and Thinkers

📚 [Paper](https://arxiv.org/abs/2507.02092) | 🌐 [Website](https://energy-based-transformers.github.io/) | 📝 [Blog](https://alexiglad.github.io/blog/2025/ebt/) | 🧾 [Bibtex](https://github.com/alexiglad/ebt?tab=readme-ov-file#Citation) | ✏️ [Simple Training Loop](https://github.com/alexiglad/EBT/blob/main/example_code/minimal_nlp_training_loop.py)

<img src="assets/model.png" alt="Autoregressive Model Architecture" width="100%" />


Energy-Based Transformers (EBTs) are a new approach enabling **generalizable** reasoning/System 2 Thinking on any problem/modality. We demonstrate the scalability of EBTs by becoming the first approach to **outscale** feed-forward Transformers across modalities and with respect to several axes including data, depth, parameters, FLOPs, etc. EBTs can think over every single prediction being made (i.e. every token in language modeling) and tend to **generalize better** than existing models.



## Setup

To set up the environment using Conda (recommended):
```
conda create -n ebt python=3.12
conda activate ebt
pip install -r requirements.txt
```

You may want to set the $HF_HOME env variable so that your data/models cache to a desired directory and posssibly the $HF_TOKEN env variable

[Login](https://docs.wandb.ai/ref/cli/wandb-login) to wandb using `wandb login` inside of that environment.

If there are issues with PyTorch or any other packages you may also use the `gh200_requirements.txt` or `loose_requirements.txt` for requirements for GH200s and without nvidia, pytorch, and triton packages respectively. You can also create a conda environment using the `environment.yml`

For the video dataset setup please see the README at [/data/vid/](/data/vid/README.md) for dataset installation and FFPROBE installation; similarly for video inference setup please see the README at [/inference/vid/](/inference/vid/README.md).

 

## Running Code

Start by running a job script. There are two ways to do this:

##### Running a bash script directly (quick start):

```
bash job_scripts/nlp/pretrain/ebt_s1.sh
```

##### Running a bash script using slurm executor (recommended on HPC if slurm is installed):

```
bash slurm_executor.sh reference_a100 job_scripts/nlp/pretrain/ebt_s1.sh
```

- This method has a mandatory param (in this case `reference_a100`) which tells slurm_executor.sh how to build the slurm script (**Note**, *you need to tailor this and set the corresponding script according to your cluster.*). The available parameters are currently "reference_a100" (for your reference :). 
- You can also just add a slurm header to the existing bash scripts and execute scripts using sbatch, which is more standard, but this `slurm_executor.sh` is super helpful for keeping code modular
  - To add an HPC config type for slurm_executor.sh please see the reference script [job_scripts/slurm_headers/reference_a100.slurm](job_scripts/slurm_headers/reference_a100.slurm) and add the script name to [slurm_executor.sh](./slurm_executor.sh)

The key parameters in these job scripts are *the RUN_NAME, MODEL_NAME, and MODEL_SIZE*. Make sure to ctrl/cmd d (edit 3 things at once) when changing these to change the log names as well in addition to the RUN_NAME. The model size *magically* automatically sets the numbers of layers, attention heads, embed dim, etc. :) Also make sure you set the wandb run information properly (entity and project).

If you want to do multinode, you may need to set `ntasks = ngpus` and run the code using `srun python filename.py` (see the [job_scripts/nlp/pretrain/ebt_s1_mn.sh](job_scripts/nlp/pretrain/ebt_s1_mn.sh) file. Note that very little multinode training was used for this paper, hence the lack of exploration of multinode code in the codebase. *You may also need to disable GPU binding in slurm headers (i.e. dont have `#SBATCH --gpu-bind=verbose,closest`)*, more on that [here](https://lightning.ai/docs/pytorch/stable/clouds/cluster_advanced.html).

##### Run example minimalistic training loop for Transformer++ vs EBT language models (NOTE this doesn't reproduce paper results)

```
python example_code/minimal_nlp_training_loop.py
```

### Energy-Based Transformers Pretraining
The pretraining scripts are located under the specific modality desired (i.e. NLP, IMG) in the `job_scripts` dir. For example, the NLP pretraining script in [job_scripts/nlp/pretrain/ebt_s1.sh](job_scripts/nlp/pretrain/ebt_s1.sh) is the System 1 EBT used for many experiments in the paper.
  - There are several similar scripts for System 2 EBTs and for other modalities in the  `job_scripts` folder, that can all be run in the same manner described above (with bash directly or slurm).
  - If you are training your own EBT, I recommend starting with these System 1 hyperparameters for the respective modality and then tweaking them from there.


### Energy-Based Transformers Inference

The inference scripts are located under the specific modality desired and under the inference subdirectory. For example, the NLP inference script for EBTs in [job_scripts/nlp/inference/ebt.sh](job_scripts/nlp/inference/ebt.sh) is useful. The biggest difference between pretraining and inference is the use of a pretrained checkpoint with `--only_test_model_ckpt`, `--only_test` for telling the trained to not train and just test, as well as `--execution_mode "inference"` which controls the "mode" of the training loop. Everything else is relatively standard and can be used to do more complex inference procedures, such as EBT System 2 Thinking with self-verification (`infer_generated_samples` and `infer_ebt_advanced`)
  - If you are looking for more fine-grained control over these it's possible to use `--only_test` without using `--execution_mode "inference"` (for example if all you want to do is calculate the perplexity of language models). You can see more on this in the `test_step` of `base_model_trainer.py` as well as in `train_model.py`
  - Most of the other hparams in the script dont matter as they will be inherited from the trained ckpt (this logic is in `train_model.py`).
  - If you are using these inference scripts make sure to fill in the `your/model/ckpt` as an actual .ckpt file!


## General Code Flow

- The executed job script will send all the hparams to `train_model.py`. `train_model.py` will do things like set up distributed training, determine the number of GPUs, determine model config/size, and alter the behavior based off the specified hparams/args.
- `train_model.py` will usually call `base_model_trainer.py`---this is the pytorch lightning trainer that is responsible for all training loop behavior, validation behavior, testing, dataset setup, logging, creating the optimizer and lr scheduler, etc. Feel free to check it out.
  - The most important lines in base_model_trainer.py are the `eval_step` function as well as the instantiation of `self.model`.
  - This file is also where the datasets are instantiated, particularly the `setup` function.
- After the instantiation of `self.model`, different models will be created. Some examples of these can be seen in `model/nlp`. These are also pytorch lightning modules, that way lightning handles all the gpu calls, distributed things, etc.
- After this, `train_model.py` will do something such as call `trainer.fit()` which will actually start training!
- Generally, you should need to change little code in train_model.py and base_model_trainer.py, aside from maybe adding some new args, models, metrics, and datasets. The biggest changes usually need to be made when adding new models and their architectures
- If all you want is to get the model code and put it in your own training loop, just refer to the `model/` directory which has the model architectures, model forward/loss calculations, and other miscellaneous utils that models use.
- For more details on the code please reference [CODE_INFO.md](./CODE_INFO.md).


## Repo Structure

```
┌── abbreviations.md                  # has various abbreviations used in the repo
├── base_model_trainer.py             # 2nd most important file, contains PL training loop
├── CODE_INFO.md                      # some extra information on coding
├── data
│   ├── img                           # various image datasets
│   ├── nlp                           # various NLP datasets
│   └── vid                           # various video datasets
├── example_code
│   └── minimal_nlp_training_loop.py  # minimal training loop for language modeling
├── inference
│   ├── img                           # inference code for images
│   ├── nlp                           # inference code for NLP
│   └── vid                           # inference code for video
├── job_scripts # all the bash/slurm scripts for training/running jobs
├── model
│   ├── ar_ebt_adaln.py               # code for autoregressive EBT with adaptive layer norm, based on llama2
│   ├── ar_ebt_default.py             # code for autoregressive EBT default, based on llama2
│   ├── ar_ebt_time_embed.py          # code for autoregressive EBT with time embedding, based on llama2
│   ├── ar_transformer.py             # baseline Transformer++ from llama2
│   ├── bi_ebt_adaln.py               # bidirectional EBT with adaptive layer norm, based on DiT
│   ├── diffusion                     # folder from diffusion repo
│   ├── diffusion_transformer.py      # code from DiT repo
│   ├── img                           # models for image denoising, generation, etc
│   ├── model_utils.py                # useful code that several models share
│   ├── nlp                           # nlp model implementations
│   ├── replay_buffer.py              # causal replay buffer, only used by System 2 EBTs
│   └── vid                           # video model implementations
├── optimization.py                   # some additional optimization code, LR scheduler, etc
├── slurm_executor.sh                 # helper code for executing slurm scripts
├── train_model.py                    # most important file, argparses and sets up PL trainer, distributed, etc
└── utils                             # various utilization files
```


A more thorough structure tree of every file is also in [CODE_INFO.md](./CODE_INFO.md).

## Additional Resources
- [JVP Flash Attention Implementation](https://github.com/amorehead/jvp_flash_attention), enabling flash attention with 2nd order derivatives (HVPs)
- [NanoEBM](https://github.com/sdan/nanoEBM?tab=readme-ov-file)

## Citation

If you find this repository useful, please consider giving a star ⭐ and citation 🙃:

```bibtex
@misc{gladstone2025energybasedtransformersscalablelearners,
  title={Energy-Based Transformers are Scalable Learners and Thinkers}, 
  author={Alexi Gladstone and Ganesh Nanduru and Md Mofijul Islam and Peixuan Han and Hyeonjeong Ha and Aman Chadha and Yilun Du and Heng Ji and Jundong Li and Tariq Iqbal},
  year={2025},
  eprint={2507.02092},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2507.02092}, 
}
```

## Contact

If you have questions feel free to post them on github issues, hugging face daily papers, or email me ([alexigladstone@gmail.com](mailto:alexigladstone@gmail.com)). Thank-you so much and hope you enjoy!
