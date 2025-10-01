import torch
import os
import h5py
import torch
import numpy as np
from diffusers import AutoencoderKL
from model.model_utils import get_encoded_images


from torch.utils.data import DataLoader
from tqdm import tqdm

def encode_dataset_features(hparams, encoder, dataset, hdf5_file_path, split):
	print(f"Encoding {len(dataset)} videos in {split} split to store at {hdf5_file_path}")

	device = "cuda" if torch.cuda.is_available() else "cpu"

	# configured for tdv models
	encoder.to(device)
	encoder.eval()

	# freeze encoder params
	for p in encoder.parameters():
		p.requires_grad = False

	from torch.nn.utils.rnn import pad_sequence
	def collate_fn_pad(batch):
		frames, video_ids = zip(*batch)  # videos: list of [T, C, H, W] tensors

		# Pad sequences along time dimension
		lengths = [v.shape[0] for v in frames]
		padded_videos = pad_sequence(frames, batch_first=True)  # shape: [B, max_T, C, H, W]

		return padded_videos, lengths, list(video_ids)

	dataloader = DataLoader(dataset, batch_size=hparams.batch_size_per_device, shuffle=False, pin_memory=True, drop_last=False, num_workers=hparams.num_workers, collate_fn=collate_fn_pad)
	
	with h5py.File(hdf5_file_path, "a") as h5f:
		split_group = h5f.require_group(split)

		for batch in tqdm(dataloader, desc="Preencoding dataset features: "):
			frames, lengths, video_ids = batch																									
			batch_size, max_num_frames, c, h, w = frames.shape

			with torch.no_grad():
				frames = frames.to(device)
				if hparams.backbone_type == "dinov2":
					encoded_frames_output = encoder(frames.reshape(-1, c, h, w), is_training=True)

					encoded_class_tokens = encoded_frames_output['x_norm_clstoken']																	# [batch_size * num_frames, dim]
					encoded_patch_tokens = encoded_frames_output['x_norm_patchtokens'] 																# [batch_size * num_frames, num_patches, dim]

					encoded_class_tokens = encoded_class_tokens.reshape(batch_size, max_num_frames, -1)
					encoded_patch_tokens = encoded_patch_tokens.reshape(batch_size, max_num_frames, -1, encoded_class_tokens.shape[-1])

					encodings = torch.cat((encoded_class_tokens.unsqueeze(2), encoded_patch_tokens), dim=2)
				elif hparams.backbone_type == "mae":
					num_patches = (h//16) * (w//16)		#TODO: make this generic							
					noise = torch.zeros((batch_size*max_num_frames, num_patches), dtype=torch.float32)

					# -- mae returns output in [seq_len, num_patches, dim]
					encodings = encoder(frames.reshape(-1, c, h, w), noise=noise, return_dict=True).last_hidden_state					
					encodings = encodings.reshape(batch_size, max_num_frames, -1, encodings.shape[-1])												# [batch_size, num_frames, num_patches, dim]
				elif hparams.backbone_type == "vae":
					encodings = get_encoded_images(frames.reshape(-1, c, h, w), "vae", encoder, sdxl_vae_standardization = hparams.sdxl_vae_standardization)
					encodings = encodings.reshape(batch_size, max_num_frames, -1)

			# store features in hdf5 file
			encodings = encodings.cpu().numpy()

			for i in range(batch_size):
				video_id = str(video_ids[i])
				if video_id not in split_group:
					split_group.create_dataset(video_id, data=encodings[i, :lengths[i]])
				# else:
				# 	print(f"SKIPPING VIDEO ID BECAUSE IT ALREADY EXISTS: {video_id}")

	print(f"Encoded {len(dataset)} videos in {split} split and stored in {hdf5_file_path}")