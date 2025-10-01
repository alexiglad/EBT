import torch
from torch import nn
import pytorch_lightning as L
import traceback
import math
import random

from model.model_utils import *
from model.replay_buffer import CausalReplayBuffer

class EBT_VID(L.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        if isinstance(hparams, dict):#passed in from model ckpt
            self.hparams.update(hparams)
        else:
            self.hparams.update(vars(hparams))
        
        if(self.hparams.energy_loss_fn=='l1_loss'):
            self.energy_loss_fn = nn.L1Loss()
        elif(self.hparams.energy_loss_fn=='smooth_l1_loss'):
            self.energy_loss_fn = nn.SmoothL1Loss()
        else:#default
            self.energy_loss_fn = nn.MSELoss()

        self.alpha = nn.Parameter(torch.tensor(float(self.hparams.mcmc_step_size)), requires_grad=self.hparams.mcmc_step_size_learnable)
        self.langevin_dynamics_noise_std = nn.Parameter(torch.tensor(float(self.hparams.langevin_dynamics_noise)), requires_grad=False) # if using self.hparams.langevin_dynamics_noise_learnable this will be turned on in warm_up_finished func
        self.finished_warming_up = False

        self.image_encoder = load_image_encoder(self.hparams.backbone_type, self.hparams.vit_backbone_size) # dtype 32 by default; also image_encoder here can refer to an entire VAE
        if self.hparams.backbone_type == "vae" and self.hparams.embedding_dim != 3136: # hardcoded original dim of vae from sd-vae
            self.encoder_down_projection = nn.Linear(3136, self.hparams.embedding_dim, bias=True)
            init_whole_model_weights(self.encoder_down_projection, self.hparams.weight_initialization_method, weight_initialization_gain=self.hparams.weight_initialization_gain)
            self.encoder_dim = 3136
        else:
            self.encoder_down_projection = nn.Identity() # dont downproject, encoder dim = embed dim
            self.encoder_dim = self.hparams.embedding_dim

        for param in self.image_encoder.parameters():
            param.requires_grad = False

        self.transformer = setup_ebt(self.hparams)

        self.mcmc_replay_buffer = 'mcmc_replay_buffer' in self.hparams and self.hparams.mcmc_replay_buffer and self.hparams.execution_mode != "inference"
        if self.mcmc_replay_buffer:
            replay_buffer_max_size = self.hparams.mcmc_replay_buffer_size
            self.replay_buffer_samples = self.hparams.batch_size_per_device * self.hparams.mcmc_replay_buffer_sample_bs_percent
            self.replay_buffer = CausalReplayBuffer(max_size=replay_buffer_max_size, sample_size=self.replay_buffer_samples)
        
        self.reconstruction_criterion = nn.SmoothL1Loss(beta=1.0)
        
        # DEBUGGING CODE ################################################################################################################################################
        if self.hparams.debug_unused_parameters:
            self.used_parameters = set()
            self.parameters_not_to_check = set() # dont check these since may be frozen or dont want them to update
            
    
    def forward(self, original_real_embeddings_input, start_pos = 0, learning = True, replay_buffer_embeddings = None, no_randomness = True): # accepts the real embeddings as input (B, S, D); a lot of the logic here is just for S2 params, see pseudocode in paper for a more concise view of how this works. it can be < 10 LOC
        # original here is to disdinguish between the original space of the encoder and the embedding space of the trans, these may be the same in some cases (see logic in init, sometimes i use nn.Identity)
        predicted_embeddings_list = []
        predicted_energies_list = []
        batch_size = original_real_embeddings_input.shape[0]
        seq_length = original_real_embeddings_input.shape[1]

        alpha = torch.clamp(self.alpha, min=0.0001)
        if not no_randomness and self.hparams.randomize_mcmc_step_size_scale != 1:
            expanded_alpha = alpha.expand(batch_size, seq_length, 1)

            scale = self.hparams.randomize_mcmc_step_size_scale
            low = alpha / scale
            high = alpha * scale
            alpha = low + torch.rand_like(expanded_alpha) * (high - low)

        langevin_dynamics_noise_std = torch.clamp(self.langevin_dynamics_noise_std, min=0.000001)

        real_embeddings_input = self.encoder_down_projection(original_real_embeddings_input)

        original_predicted_embeddings = self.corrupt_embeddings(original_real_embeddings_input) # B, S, D ; is called predicted bc initial prediction may be poor
        if replay_buffer_embeddings is not None: # using replay buffer, use the logits instead of corruption
            original_predicted_embeddings[batch_size - replay_buffer_embeddings.shape[0]:] = replay_buffer_embeddings # NOTE this assumes the fresh data is concatted first

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
                    original_predicted_embeddings.requires_grad_().reshape(batch_size, seq_length, self.encoder_dim) # B, S, D
                else: # default, do detach
                    original_predicted_embeddings = original_predicted_embeddings.detach().requires_grad_().reshape(batch_size, seq_length, self.encoder_dim) # B, S, D
                
                predicted_embeddings = self.encoder_down_projection(original_predicted_embeddings)
                
                if self.hparams.langevin_dynamics_noise != 0: # only use langevyn dynamics once model is warmed up
                    ld_noise = torch.randn_like(predicted_embeddings.detach(), device=predicted_embeddings.device) * langevin_dynamics_noise_std # langevin dynamics
                    predicted_embeddings = predicted_embeddings + ld_noise
                
                all_embeddings = torch.cat((real_embeddings_input, predicted_embeddings), dim = 1) # B, 2*S, D
                
                energy_preds = self.transformer(all_embeddings, start_pos = start_pos, mcmc_step=mcmc_step) # is B, 2*S, D; checked and there are no in place ops; mcmc_step only applies to when using certain types of ebt
                energy_preds = energy_preds.reshape(-1, 1)
                predicted_energies_list.append(energy_preds)

                if self.hparams.truncate_mcmc:  #retain_graph defaults to create_graph value here; if learning is true then create_graph else dont (inference)
                    if i == (len(mcmc_steps) - 1):
                        predicted_embeds_grad = torch.autograd.grad([energy_preds.sum()], [original_predicted_embeddings], create_graph=learning)[0]
                    else:
                        predicted_embeds_grad = torch.autograd.grad([energy_preds.sum()], [original_predicted_embeddings], create_graph=False)[0]
                else:
                    predicted_embeds_grad = torch.autograd.grad([energy_preds.sum()], [original_predicted_embeddings], create_graph=learning)[0]
                
                if self.hparams.clamp_futures_grad:
                    min_and_max = self.hparams.clamp_futures_grad_max_change / (self.alpha)
                    # predicted_embeds_grad = scale_clamp(predicted_embeds_grad, -min_and_max, min_and_max)
                    predicted_embeds_grad = torch.clamp(predicted_embeds_grad, min = -min_and_max, max = min_and_max)

                if torch.isnan(predicted_embeds_grad).any() or torch.isinf(predicted_embeds_grad).any():
                    raise ValueError("NaN or Inf gradients detected during MCMC.")
                    
                original_predicted_embeddings = original_predicted_embeddings - alpha * predicted_embeds_grad
                
                predicted_embeddings_list.append(original_predicted_embeddings)

        return predicted_embeddings_list, predicted_energies_list
    
    def forward_loss_wrapper(self, x, phase="train"):
        no_randomness = False if phase == "train" else True
        learning = (phase == "train")
        #real embeddings here are embeddings extracted from the real video, predicted_embeddings are the predictions (initial pred is often random)   
        if self.hparams.preencode_video and os.path.exists(self.hparams.hdf5_file_path):
            # print(f"Using pre-computed embeddings!")
            if self.hparams.backbone_type == "dinov2":
                real_embeddings = x[:, :, 0, :]
            elif self.hparams.backbone_type == "vae":
                real_embeddings = x

        else:
            batch_size = x.shape[0]
            seq_length = x.shape[1]

            x = x.reshape(-1, *x.shape[2:]) # B*(S+1), C, W, H

            real_embeddings = get_encoded_images(x, self.hparams.backbone_type, self.image_encoder, sdxl_vae_standardization = self.hparams.sdxl_vae_standardization)
            # print(f"real embeddings shape before reshape: {real_embeddings.shape}")
            real_embeddings = real_embeddings.reshape(batch_size, seq_length, self.encoder_dim) # B, S+1, D
            # print(f"real embeddings shape after reshape: {real_embeddings.shape}")

        if not no_randomness and self.mcmc_replay_buffer: # dont do this when doing val/testing
            real_embeddings_input, replay_buffer_embeddings, real_embeddings_gt = self.replay_buffer.get_batch(real_embeddings) # this automatically does indexing for inputs and gt next embeddings while also passing back the replay buffer embeddings
            predicted_embeddings_list, predicted_energies_list = self(real_embeddings_input, learning=learning, replay_buffer_embeddings = replay_buffer_embeddings, no_randomness = no_randomness)
            self.replay_buffer.update(real_embeddings.detach(), predicted_embeddings_list[-1].detach()) # update using the final predicted embeddings
        else:
            real_embeddings_input = real_embeddings[:, :-1, :] #B, S, D; similar to predictions but is all but last frame since need first n frames to be real and last n frames to be predicted
            predicted_embeddings_list, predicted_energies_list = self(real_embeddings_input, learning=learning, no_randomness = no_randomness)
            real_embeddings_gt =  real_embeddings[:, 1:, :] #B, S, D; for rec loss when comparing to predicted_embeddings so compare frame by frame, extract in same way

        reconstruction_loss = 0.0
        final_reconstruction_loss = 0.0
        energy_loss = 0.0
        out_of_bounds_loss = 0.0

        total_mcmc_steps = len(predicted_energies_list)
        for mcmc_step, (predicted_embeddings, predicted_energy) in enumerate(zip(predicted_embeddings_list, predicted_energies_list)):
            #loss calculations
            if self.hparams.truncate_mcmc:
                if mcmc_step == (total_mcmc_steps - 1):
                    reconstruction_loss = self.reconstruction_criterion(predicted_embeddings, real_embeddings_gt)
                    final_reconstruction_loss = reconstruction_loss.detach()
            else:
                reconstruction_loss += self.reconstruction_criterion(predicted_embeddings, real_embeddings_gt)
                if mcmc_step == (total_mcmc_steps - 1):
                    final_reconstruction_loss = self.reconstruction_criterion(predicted_embeddings, real_embeddings_gt).detach()
                    reconstruction_loss = reconstruction_loss / total_mcmc_steps # normalize so is indifferent to number of mcmc steps
            
            #pure logging things (no function for training)
            if mcmc_step == 0:
                initial_loss = self.reconstruction_criterion(predicted_embeddings, real_embeddings_gt).detach()
                initial_pred_energies = predicted_energy.squeeze().mean().detach()
            if mcmc_step == (total_mcmc_steps - 1):
                final_pred_energies = predicted_energy.squeeze().mean().detach()

            # NOT USED ANYMORE LOSSES ##################
            if self.hparams.energy_loss_coeff != 0.0:
                energy_labels = self.calc_distance(predicted_embeddings, real_embeddings_gt)
                if self.hparams.energy_loss_hinge == 0.0: # for doing margin based energy regressive pred, helps with inherent randomness in metric used for REBM
                    energy_loss += self.energy_loss_fn(predicted_energy.squeeze(), energy_labels.reshape(-1)) # labels do not have grad, preds do; both after ops here will have (B*(S-1))
                else:
                    energy_loss += hinged_mse_loss(predicted_energy.squeeze(), energy_labels.reshape(-1), margin = self.hparams.energy_loss_hinge)
                    
            if self.hparams.out_of_bounds_loss_coeff != 0.0:
                out_of_bounds_loss += calc_out_of_bounds_loss(predicted_energy)
            ######################################################
                                        
        initial_final_pred_energies_gap = initial_pred_energies - final_pred_energies
        
        total_loss = self.hparams.energy_loss_coeff * energy_loss +  self.hparams.reconstruction_coeff * reconstruction_loss + self.hparams.out_of_bounds_loss_coeff * out_of_bounds_loss
        #NOTE when returning losses make sure to detach things from comp graph
        if isinstance(energy_loss, torch.Tensor): #these are just in case are just using one or the other
            energy_loss = energy_loss.detach()
        if isinstance(reconstruction_loss, torch.Tensor):
            reconstruction_loss = reconstruction_loss.detach()
        if isinstance(out_of_bounds_loss, torch.Tensor):
            out_of_bounds_loss = out_of_bounds_loss.detach()
        
        log_dict = {
            'loss': total_loss,
            'initial_loss' : initial_loss,
            'final_step_loss': final_reconstruction_loss,
            'initial_final_pred_energies_gap': initial_final_pred_energies_gap,
        }
        return log_dict
        
    
    def corrupt_embeddings(self, embeddings): 
        if self.hparams.denoising_initial_condition == "most_recent_embedding":
            predicted_embeddings = embeddings.clone()
        elif self.hparams.denoising_initial_condition == "random_noise":
            predicted_embeddings = torch.randn_like(embeddings)
        elif self.hparams.denoising_initial_condition == "zeros":
            predicted_embeddings = torch.zeros_like(embeddings) # default just condition on nothing. found it did not produce as good of reps. as random_noise
        else:
            raise ValueError(f"{self.hparams.denoising_initial_condition} denoising_initial_condition not yet supported")
        return predicted_embeddings
    
    def calc_distance(self, pred_embeddings, gt_embeddings): # NOTE not used is old code for an old loss calc
        with torch.set_grad_enabled(False): # dont want grad since dont want model to see how energy was calculated
            # both embed tensors have shape (B, (S-1), D)

            if self.hparams.embeddings_distance_fn == 'euclidean':
                raw_distance = torch.norm(gt_embeddings - pred_embeddings, dim=2, p=2)
            elif self.hparams.embeddings_distance_fn == 'manhattan':
                raw_distance = torch.norm(gt_embeddings - pred_embeddings, dim=2, p=1)
            elif self.hparams.embeddings_distance_fn == 'cosine':
                cos_sim = torch.nn.functional.cosine_similarity(gt_embeddings, pred_embeddings, dim=2)
                raw_distance = (-1 * cos_sim) + 1  # Convert similarity to distance
            else:
                raise ValueError(f"Invalid distance function specified: {self.hparams.embeddings_distance_fn}")

            # Average raw distance across the sequence length and make sure is non negative due to rounding
            min_val = torch.min(raw_distance).detach()
            adj_distance = torch.where(raw_distance < 0, raw_distance - min_val, raw_distance)
            if torch.any(adj_distance < 0):
                raise ValueError("no values should be less than zero, adjust above code")

            # Apply normalization transformations if needed
            if self.hparams.embeddings_distance_fn == 'normalized_euclidean':
                mean_distance = torch.mean(adj_distance)
                std_distance = torch.std(adj_distance)
                adj_distance = (adj_distance - mean_distance) / (std_distance + 1e-6)
            elif self.hparams.embeddings_distance_fn == 'cosine':
                if self.hparams.scale_cosine_sim_decay != 0: # just a way of scaling cosine sim that I found correlates to a good energy empirically
                    adj_distance = 1 - torch.exp(-1 * self.hparams.scale_cosine_sim_decay * adj_distance)
            return adj_distance
        
    def warm_up_finished(self):
        self.finished_warming_up = True
        self.langevin_dynamics_noise_std.requires_grad = self.hparams.langevin_dynamics_noise_learnable
    
    def ebt_advanced_inference(self, original_real_embeddings_input, start_pos=0, learning=True):
        raise NotImplementedError("have not yet implemented this code")