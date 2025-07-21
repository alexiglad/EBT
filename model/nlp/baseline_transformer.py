import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as L
import torch.optim as optim
from torchmetrics import Accuracy

from transformers import AutoTokenizer

import math
import random
import os
from model.model_utils import *


class Baseline_Transformer_NLP(L.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        if isinstance(hparams, dict):#passed in from model ckpt
            self.hparams.update(hparams)
        else:
            self.hparams.update(vars(hparams))
        
        tokenizer = AutoTokenizer.from_pretrained(self.hparams.tokenizer, clean_up_tokenization_spaces = False)
        self.tokenizer_pad_token_id = tokenizer.eos_token_id # is token 0, was right padding things
        
        self.vocab_size = len(tokenizer) # self.vocab_size = tokenizer.vocab_size caused errors since is smaller than len(tokenizer), is 50254 for neox-20b, len tokenizer is 50277 so decided to use that
        self.embeddings = nn.Embedding(self.vocab_size, self.hparams.embedding_dim)
        init_whole_model_weights(self.embeddings, self.hparams.weight_initialization_method, weight_initialization_gain=self.hparams.weight_initialization_gain)
        
        self.log_softmax = nn.LogSoftmax(dim = -1)

        self.transformer = setup_transformer(self.hparams)
        
        self.output = nn.Linear(self.hparams.embedding_dim, self.vocab_size, bias = False)
        init_whole_model_weights(self.output, self.hparams.weight_initialization_method, weight_initialization_gain=self.hparams.weight_initialization_gain)
        
        self.finished_warming_up = False
        

    def forward(self, x, start_pos = 0, learning = True, return_raw_logits = False): # accepts input_ids as input
        embeddings = self.embeddings(x) # x here is input_ids
        
        predicted_embeddings = self.transformer(embeddings, start_pos = start_pos, learning = learning) # BS, S, D
        predicted_logits = self.output(predicted_embeddings) #BS, S, vocab_size
        if return_raw_logits:
            return predicted_logits
        else:
            predicted_distribution = self.log_softmax(predicted_logits).reshape(-1, self.vocab_size) # BS*S, V; reshape since preds for nll should be 2d
            return predicted_distribution
        

    def forward_loss_wrapper(self, x, phase="train"):
        input_ids = x['input_ids'].squeeze(dim=1)[:, :-1] # x['input_ids'] shape is BS, S+1, only input first S--remove next tokens

        predicted_distribution = self(input_ids)
        
        next_token_indices = x['input_ids'].squeeze(dim=1)[:, 1:] # squeeze was to remove 1 on 2nd dim
        if self.hparams.execution_mode == "finetune": # Only tokens after "[[Answer]]: " will be calculated in finetune
            next_token_indices = mask_q_tokens(next_token_indices, self.tokenizer)
        next_token_indices = next_token_indices.reshape(-1) # BS * S; reshape since targets are supposed to be 1D
        
        cce_loss = F.nll_loss(predicted_distribution, next_token_indices, ignore_index=self.tokenizer_pad_token_id)
        ppl_loss = torch.exp(cce_loss).detach()

        # Count non-padded tokens for accurate perplexity calculation
        num_tokens = (next_token_indices != self.tokenizer_pad_token_id).sum().item()

        log_dict = {
            'loss': cce_loss,
            'perplexity': ppl_loss,
            'batch_total_loss': cce_loss * num_tokens,  # For accurate perplexity
            'batch_num_tokens': num_tokens  # For accurate perplexity
        }
        return log_dict