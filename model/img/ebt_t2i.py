import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as L
import torch.optim as optim
from torchmetrics import Accuracy
import traceback
from torchvision.transforms import functional as TF
import torchvision.models as models
import math
import random
from diffusers import AutoencoderKL

from model.model_utils import *
from model.diffusion import create_diffusion
from model.diffusion.gaussian_diffusion import mean_flat
# from model.replay_buffer import CausalReplayBuffer # fix



class EBT_IMG_T2I(L.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        if isinstance(hparams, dict):#passed in from model ckpt
            self.hparams.update(hparams)
        else:
            self.hparams.update(vars(hparams))

        self.image_encoder = load_image_encoder(self.hparams.backbone_type, self.hparams.vit_backbone_size, device = self.device, use_ema=True) # dtype 32 by default, for here is actually a VAE
        self.text_encoder, self.text_encoder_dim = get_clip_text_encoder(self.hparams.clip_text_encoder_size)
        self.text_embeddings_projector = nn.Linear(self.text_encoder_dim, self.hparams.embedding_dim, bias=True)

        self.alpha = nn.Parameter(torch.tensor(float(self.hparams.mcmc_step_size)), requires_grad=self.hparams.mcmc_step_size_learnable)
        self.langevin_dynamics_noise_std = nn.Parameter(torch.tensor(float(self.hparams.langevin_dynamics_noise)), requires_grad=False) # if using self.hparams.langevin_dynamics_noise_learnable this will be turned on in warm_up_finished func
        self.finished_warming_up = False

        for param in self.image_encoder.parameters():
            param.requires_grad = False
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        self.transformer = setup_bidirectional_ebt(self.hparams)

        self.reset_image_encoder_decoder = False # this is bc of a bug with the vae requiring reloading to use... i think its some interact with PL

        self.smooth_l1 = nn.SmoothL1Loss(beta=1.0)
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()


        # note that this works on a fixed image shape (non dynamic) so we define self.raw_embeddings_shape below for helping
        self.raw_embeddings_shape = [self.transformer.in_channels, self.transformer.channel_height, self.transformer.channel_width] # shape of just embeddings no batch dim

        self.mcmc_replay_buffer = 'mcmc_replay_buffer' in self.hparams and self.hparams.mcmc_replay_buffer and self.hparams.execution_mode != "inference"
        if self.mcmc_replay_buffer:
            replay_buffer_max_size = self.hparams.mcmc_replay_buffer_size
            self.replay_buffer_samples = self.hparams.batch_size_per_device * self.hparams.mcmc_replay_buffer_sample_bs_percent
            raise NotImplementedError("need to implement bidirectional replay buffer")
            # self.replay_buffer = CausalReplayBuffer(max_size=replay_buffer_max_size, sample_size=self.replay_buffer_samples)

    def forward(self, y, start_pos = 0, learning = True, initialize_predictions = None, no_randomness = True): # y here is caption_embeddings since is just text conditional
        predicted_embeddings_list = []
        predicted_energies_list = []
        batch_size = y.shape[0]
        projected_caption_embeddings = self.text_embeddings_projector(y)

        condition = projected_caption_embeddings

        alpha = torch.clamp(self.alpha, min=0.0001)
        if not no_randomness and self.hparams.randomize_mcmc_step_size_scale != 1:
            expanded_alpha = alpha.expand(batch_size, 1, 1, 1)

            scale = self.hparams.randomize_mcmc_step_size_scale
            low = alpha / scale
            high = alpha * scale
            alpha = low + torch.rand_like(expanded_alpha) * (high - low)

        langevin_dynamics_noise_std = torch.clamp(self.langevin_dynamics_noise_std, min=0.000001)

        predicted_embeddings = self.corrupt_embeddings(projected_caption_embeddings) # B, C, W, H
        if initialize_predictions is not None:
            predicted_embeddings[batch_size - initialize_predictions.shape[0]:] = initialize_predictions # NOTE this assumes the fresh data is concatted first

        mcmc_steps = [] # in the general case of no randomize_mcmc_num_steps then this has len == self.hparams.randomize_mcmc_num_steps
        for step in range(self.hparams.mcmc_num_steps):
            if not no_randomness and hasattr(self.hparams, 'randomize_mcmc_num_steps') and self.hparams.randomize_mcmc_num_steps > 0:
                if self.hparams.randomize_mcmc_num_steps_final_landscape: # makes so only applies rand steps to final landscape
                    if step == (self.hparams.mcmc_num_steps - 1):
                        min_steps = 1 if self.hparams.randomize_mcmc_num_steps_min == 0 else self.hparams.randomize_mcmc_num_steps_min
                        repeats = torch.randint(min_steps, self.hparams.randomize_mcmc_num_steps + 2, (1,)).item()
                        mcmc_steps.extend([step] * repeats)
                    else:
                        mcmc_steps.append(step)
                else:
                    min_steps = 1 if self.hparams.randomize_mcmc_num_steps_min == 0 else self.hparams.randomize_mcmc_num_steps_min
                    repeats = torch.randint(min_steps, self.hparams.randomize_mcmc_num_steps + 2, (1,)).item()
                    mcmc_steps.extend([step] * repeats)
            elif no_randomness and hasattr(self.hparams, 'randomize_mcmc_num_steps') and self.hparams.randomize_mcmc_num_steps > 0: # use max steps
                if step == (self.hparams.mcmc_num_steps - 1): # i found this was a better pretraining metric and was more stable, only do several steps on final energy landscape instead of over all energy landscapes
                    mcmc_steps.extend([step] * (self.hparams.randomize_mcmc_num_steps + 1))
                else:
                    mcmc_steps.append(step)
            else:
                mcmc_steps.append(step)

        with torch.set_grad_enabled(True): # set to true for validation since grad would be off
            for i, mcmc_step in enumerate(mcmc_steps):
                if self.hparams.no_mcmc_detach:
                    predicted_embeddings = predicted_embeddings.requires_grad_() # B, C, W, H
                else: # default, do detach
                    predicted_embeddings = predicted_embeddings.detach().requires_grad_() # B, C, W, H

                if self.hparams.langevin_dynamics_noise != 0 and not (no_randomness and self.hparams.no_langevin_during_eval):
                    ld_noise = torch.randn_like(predicted_embeddings.detach(), device=predicted_embeddings.device) * langevin_dynamics_noise_std # langevin dynamics
                    predicted_embeddings = predicted_embeddings + ld_noise

                energy_preds = self.transformer(predicted_embeddings, condition)
                energy_preds = energy_preds.mean(dim=(-1, -2)) # B
                predicted_energies_list.append(energy_preds)

                if self.hparams.truncate_mcmc:  #retain_graph defaults to create_graph value here; if learning is true then create_graph else dont (inference)
                    if i == (len(mcmc_steps) - 1):
                        predicted_embeds_grad = torch.autograd.grad(
                            outputs=energy_preds,
                            inputs=predicted_embeddings,
                            grad_outputs=torch.ones_like(energy_preds),
                            create_graph=learning
                        )[0]
                    else:
                        predicted_embeds_grad = torch.autograd.grad(
                            outputs=energy_preds,
                            inputs=predicted_embeddings,
                            grad_outputs=torch.ones_like(energy_preds),
                            create_graph=False
                        )[0]
                else:
                    predicted_embeds_grad = torch.autograd.grad(
                        outputs=energy_preds,
                        inputs=predicted_embeddings,
                        grad_outputs=torch.ones_like(energy_preds),
                        create_graph=learning
                    )[0]

                if self.hparams.clamp_futures_grad:
                    min_and_max = self.hparams.clamp_futures_grad_max_change / (self.alpha)
                    # predicted_embeds_grad = scale_clamp(predicted_embeds_grad, -min_and_max, min_and_max)
                    predicted_embeds_grad = torch.clamp(predicted_embeds_grad, min = -min_and_max, max = min_and_max)

                if torch.isnan(predicted_embeds_grad).any() or torch.isinf(predicted_embeds_grad).any():
                    raise ValueError("NaN or Inf gradients detected during MCMC.")

                predicted_embeddings = predicted_embeddings - alpha * predicted_embeds_grad

                predicted_embeddings_list.append(predicted_embeddings)

        return predicted_embeddings_list, predicted_energies_list


    def forward_loss_wrapper(self, x, phase="train"):
        no_randomness = False if phase == "train" else True
        learning = (phase == "train")

        images = x['image']
        captions = x['caption']
        caption_embeddings = get_text_embeddings(self.text_encoder, captions)
        image_embeddings = get_encoded_images(images, self.hparams.backbone_type, self.image_encoder, sdxl_vae_standardization = True)

        if not no_randomness and self.mcmc_replay_buffer: # dont do this when doing val/testing
            caption_embeddings, replay_buffer_embeddings, image_embeddings = self.replay_buffer.get_batch(caption_embeddings)
            predicted_embeddings_list, predicted_energies_list = self(caption_embeddings, learning=learning, initialize_predictions=replay_buffer_embeddings, no_randomness = no_randomness)
            self.replay_buffer.update(predicted_embeddings_list[-1].detach()) # update using the final predicted embeddings
        else:
            predicted_embeddings_list, predicted_energies_list = self(caption_embeddings, learning=learning, no_randomness = no_randomness)

        reconstruction_loss = 0
        total_mcmc_steps = len(predicted_energies_list)
        for mcmc_step, (predicted_embeddings, predicted_energy) in enumerate(zip(predicted_embeddings_list, predicted_energies_list)):
            #loss calculations
            if self.hparams.truncate_mcmc:
                if mcmc_step == (total_mcmc_steps - 1):
                    reconstruction_loss = self.smooth_l1(predicted_embeddings, image_embeddings)
                    final_reconstruction_loss = reconstruction_loss.detach()
            else:
                reconstruction_loss += self.smooth_l1(predicted_embeddings, image_embeddings)
                if mcmc_step == (total_mcmc_steps - 1):
                    final_reconstruction_loss = self.smooth_l1(predicted_embeddings, image_embeddings).detach()
                    reconstruction_loss = reconstruction_loss / total_mcmc_steps # normalize so is indifferent to number of mcmc steps

            #pure logging things (no function for training)
            if mcmc_step == 0:
                initial_loss = self.smooth_l1(predicted_embeddings, image_embeddings).detach()
                initial_pred_energies = predicted_energy.squeeze().mean().detach()
            if mcmc_step == (total_mcmc_steps - 1):
                final_pred_energies = predicted_energy.squeeze().mean().detach()
                mse_raw_loss = torch.mean(mean_flat((image_embeddings - predicted_embeddings) ** 2)).detach()
                l1_loss = self.l1(predicted_embeddings, image_embeddings).detach()

        initial_final_pred_energies_gap = initial_pred_energies - final_pred_energies
        total_loss = self.hparams.reconstruction_coeff * reconstruction_loss

        log_dict = {
            'loss': total_loss,
            'initial_loss' : initial_loss,
            'final_step_loss': final_reconstruction_loss,
            'initial_final_pred_energies_gap': initial_final_pred_energies_gap,
            'mse_raw_loss': mse_raw_loss,
            'l1_loss': l1_loss,
        }

        is_update_step = (self.trainer.fit_loop.epoch_loop.batch_progress.current.completed + 1) % self.trainer.accumulate_grad_batches == 0 # so dont have issues with gradient accumulation
        if self.trainer.global_step % self.hparams.log_image_every_n_steps == 0 and is_update_step and phase == "train":
            if not self.reset_image_encoder_decoder: # this is done to prevent bug where loading ckpt image encoder doesnt work well, not sure why ckpt image decoder doesnt load well, maybe related to HF or PL
                    self.image_encoder = load_image_encoder(self.hparams.backbone_type, self.hparams.vit_backbone_size, use_ema=True).to(self.device)
                    self.image_encoder.eval()
                    for param in self.image_encoder.parameters():
                        param.requires_grad = False
                    self.reset_image_encoder_decoder = True

            predicted_embeddings_list, predicted_energies_list = self(caption_embeddings, no_randomness = True, learning=False) # dont do randomness when getting samples and disable higher order grad

            with torch.no_grad():
                predicted_embeddings = predicted_embeddings_list[-1].detach()
                samples_for_decoding = torch.cat((predicted_embeddings[0].unsqueeze(0), image_embeddings[0].unsqueeze(0)), dim = 0) # only use first index of generated samples for loss calc
                decoded_images = self.image_encoder.decode(samples_for_decoding.reshape(-1, *image_embeddings.shape[1:]) / 0.18215).sample # constant following SDXL VAE https://github.com/CompVis/stable-diffusion
                decoded_images = denormalize(decoded_images.unsqueeze(0), self.hparams.dataset_name, self.device, self.hparams.custom_image_normalization, True).squeeze(0)

                log_dict['generated_image'] = decoded_images[0]
                log_dict['original_image'] = decoded_images[1]

        return log_dict


    def corrupt_embeddings(self, caption):
            bs = caption.shape[0]
            if self.hparams.denoising_initial_condition == "most_recent_embedding":
                raise NotImplementedError(f"most_recent_embedding denoising_initial_condition not supported for NLP yet")
            elif self.hparams.denoising_initial_condition == "random_noise":
                predicted_tokens = torch.randn(size=(bs, self.raw_embeddings_shape[0], self.raw_embeddings_shape[1], self.raw_embeddings_shape[2]), device = self.device) * self.hparams.gaussian_random_noise_scaling
            elif self.hparams.denoising_initial_condition == "zeros":
                predicted_tokens = torch.zeros(size=(bs, self.raw_embeddings_shape[0], self.raw_embeddings_shape[1], self.raw_embeddings_shape[2]), device = self.device)
            else:
                raise NotImplementedError(f"{self.hparams.denoising_initial_condition} denoising_initial_condition not yet supported")

            return predicted_tokens

    def ebt_advanced_inference(self, caption_embeddings, start_pos=0, learning=True):
        pass
        raise NotImplementedError("have not implemented this yet")
