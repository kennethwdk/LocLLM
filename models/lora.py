# Derived from https://github.com/microsoft/LoRA
#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

r"""
    Low Ranking Adaptation for LLMs scheme.

             ┌───────────────────┐
             ┆         h         ┆
             └───────────────────┘
                       ▲
                       |
                       +
                    /     \
    ┌─────────────────┐    ╭───────────────╮     Matrix initialization:
    ┆                 ┆     \      B      /      B = 0
    ┆   pretrained    ┆      \    r*d    /       A = N(0, sigma^2)
    ┆    weights      ┆       ╰─────────╯
    ┆                 ┆       |    r    |        r - rank
    ┆   W e R^(d*d)   ┆       | ◀─────▶ |
    ┆                 ┆       ╭─────────╮
    └─────────────────┘      /     A     \
              ▲             /     d*r     \
               \           ╰───────────────╯
                \                ▲
                 \              /
                  \            /
             ┌───────────────────┐
             ┆         x         ┆
             └───────────────────┘

With LoRA (Low Ranking Adaptation: https://arxiv.org/abs/2106.09685) instead of learning weights of size d*d,
we can freeze the pretrained weights and instead learn two matrices of size d*r and r*d (they will store weight updates
for the pretrained weights): the number of parameters in this case will be reduced drastically (depending on the rank of
course) yet after multiplication of matrices d*r and r*d we will get a matrix d*d which we can sum with frozen
pretrained weights and thus finetune the model.

The goal of this approach is to move weight updates into a separate matrix which is decomposed with
two matrices of a lower rank.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Dict, List

import transformers
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaRotaryEmbedding
from transformers.models.clip.modeling_clip import CLIPAttention

from .dino import MemEffAttention as DINOAttention
import models.dino

from contextlib import contextmanager
from dataclasses import dataclass


class LoRALayer():
    def __init__(
        self, 
        r: int, 
        lora_alpha: int, 
        lora_dropout: float,
        merge_weights: bool,
    ):
        """Store LoRA specific attributes in a class.

        Args:
            r: rank of the weight update matrices. To make sense of using LoRA the rank should be smaller than the rank of
                the weights of the model.  The rank can be as low as 1: https://arxiv.org/pdf/2106.09685.pdf (section 7.2)
            lora_alpha: alpha is needed for scaling updates as alpha/r
                "This scaling helps to reduce the need to retune hyperparameters when we vary r"
                https://arxiv.org/pdf/2106.09685.pdf (section 4.1)
            lora_dropout: dropout that is applied on the input in the LoRA branch (before multiplying by matrix A)
            merge_weights: whether we want to merge pretrained weights and LoRA weight updates. This is useful if one wants to use
                finetuned model as a standalone one (without storing LoRA weights separately) plus it helps to reduce
                overhead during inference.
        """
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights


class MergedLinear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        # ↓ this part is for pretrained weights
        in_features: int, 
        out_features: int, 
        # ↓ the remaining part is for LoRA
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        **kwargs
    ):
        """LoRA wrapper around linear class that is used for calculation of q, k and v matrices.

        This class has three weight matrices:
            1. Pretrained weights are stored as `self.weight` (because of the nn.Linear inheritance)
            2. LoRA A matrix as `self.lora_A`
            3. LoRA B matrix as `self.lora_B`
        Only LoRA's A and B matrices are updated, pretrained weights stay frozen.

        Args:
            in_features: number of input features of the pretrained weights
            out_features: number of output features of the pretrained weights
            r: rank of the weight update matrices. To make sense of using LoRA the rank should be smaller than the rank of
                the weights of the model.  The rank can be as low as 1: https://arxiv.org/pdf/2106.09685.pdf (section 7.2)
            lora_alpha: alpha is needed for scaling updates as alpha/r
                "This scaling helps to reduce the need to retune hyperparameters when we vary r"
                https://arxiv.org/pdf/2106.09685.pdf (section 4.1)
            lora_dropout: dropout that is applied on the input in the LoRA branch (before multiplying by matrix A)
            enable_lora: MergeLinear class is for attention mechanism where qkv are calculated with a single weight matrix. If we
                don't want to apply LoRA for all three (query, key and value) we can set it as False. For example if we want
                to apply LoRA only to `query` and `value` but keep `key` without weight updates we should pass `[True,
                False, True]`
            fan_in_fan_out: set this to True if the layer to replace stores weight like (fan_in, fan_out).  For example, gpt-2 uses
                `Conv1D` which stores weights like (fan_in, fan_out) and hence this should be set to `True`
                https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora.py#LL53C9-L53C112
            merge_weights: whether we want to merge pretrained weights and LoRA weight updates. This is useful if one wants to use
                finetuned model as a standalone one (without storing LoRA weight separately) plus it helps to reduce
                overhead during inference.
        """
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        self.fan_in_fan_out = fan_in_fan_out

        # Actual trainable parameters
        # To better understand initialization let's imagine that we have such parameters:
        # ⚬ in_features: 128 (embeddings_size)
        # ⚬ out_features: 384 (3 * embedding_size)
        # ⚬ r: 2
        # ⚬ enable_lora: [True, False, True]
        if r > 0:
            self.lora_A = nn.Parameter(
                self.weight.new_zeros((r, in_features)))  # (4, 128)
            self.lora_B = nn.Parameter(
                self.weight.new_zeros((out_features, r))  # (256, 2)
            )
            # Scaling:
            # This balances the pretrained model`s knowledge and the new task-specific adaptation
            # https://lightning.ai/pages/community/tutorial/lora-llm/
            # So, set alpha to 1.0 to fully add LoRA. If the LoRA seems to have too much effect (i.e., overfitted), set
            # alpha to lower value. If the LoRA seems to have too little effect, set alpha to higher than 1.0. You can
            # tune these values to your needs. This value can be even slightly greater than 1.0!
            # https://github.com/cloneofsimo/lora
            self.scaling = self.lora_alpha / self.r

            # Freezing the pre-trained weight matrix
            # NOTE: the bias is tunable?
            self.weight.requires_grad = False # (384, 128)

        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        """Reset all the weights, even including pretrained ones."""
        # NOTE: why reset this weights? disable now, 这个应该是先初始化再load，所以reset是没问题的？
        # nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            # Wondering why 'a' is equal to math.sqrt(5)?: https://github.com/pytorch/pytorch/issues/15314
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        """Set the module into train or eval mode if `mode` is True of False respectively.

        For train mode (train(True)) if weights are merged we need to subtract weights updates (LoRA_A @ LoRA_B) from
        pretrained weights so we can continue training LoRA's matrices A and B and keep pretrained weights frozen.

        For eval mode (train(False)) if weights are not merged we need to add weight updates to pretrained weights in
        order to reduce computational overhead during inference.

        Args:
            mode: if True the module will be set into train mode (affects Dropout and Batchnorm), if False - eval mode.

        """
        def T(w):
            return w.T if self.fan_in_fan_out else w
        # despite being called from nn.Linear this method will put all layers into train mode, including nn.Dropout
        # of course except parameters (such as self.lora_A, self.lora_B)
        nn.Linear.train(self, mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Do the forward pass.

        If LoRA's weights are merged with pretrained ones then it's a simple matrix multiplication.
        If not, then multiply pretrained weights with input, apply LoRA on input and do summation.

        Args:
            x: input tensor of shape (batch_size, context_length, embedding_size)

        Returns:
            Output tensor of shape (batch_size, context_length, 3 * embedding_size)
        """
        previous_dtype = x.dtype
        def T(w):
            return w.T if self.fan_in_fan_out else w
        # `F.linear` automatically transposes the second argument (T(self.weight) in our case)
        result = F.linear(x, T(self.weight), bias=self.bias)  # (64, 64, 128) @ (384, 128) -> (64, 64, 384)
        if self.r > 0:
            x = x.to(self.lora_A.data.dtype)
            after_A = F.linear(self.lora_dropout(x), self.lora_A)  # (64, 64, 128) @ (4, 128) -> (64, 64, 4)
            # For F.conv1d:
            # ⚬ input: input tensor of shape (minibatch, in_channels, iW)
            # ⚬ weight: filters of shape (out_channels, in_channels/groups, kW)
            # ⚬ groups: split input into groups, in_channels should be divisible by the number of groups. Default: 1
            # presumably iW - sequence width/length, kW - kernel width
            after_B = F.linear(after_A, self.lora_B)
            result = result + after_B * self.scaling
        result = result.to(previous_dtype)
        return result


def mark_only_lora_as_trainable(model: nn.Module, bias: str = 'none') -> None:
    """Freeze all modules except LoRA's and depending on 'bias' value unfreezes bias weights.

    Args:
        model: model with LoRA layers
        bias: 
            ``"none"``: all bias weights will be frozen,
            ``"lora_only"``: only bias weight for LoRA layers will be unfrozen,
            ``"all"``: all bias weights will be unfrozen.

    Raises:
        NotImplementedError: if `bias` not in ["none", "lora_only", "all"]
    """
    # freeze all layers except LoRA's
    for n, p in model.named_parameters():
        if 'lora_' not in n:
            p.requires_grad = False
        else:
            p.data = p.data.float()
            p.requires_grad = True

    # depending on the `bias` value unfreeze bias weights
    if bias == 'none':
        return
    elif bias == 'all':
        for n, p in model.named_parameters():
            if 'bias' in n:
                p.requires_grad = True
    elif bias == 'lora_only':
        for m in model.modules():
            if isinstance(m, LoRALayer) and \
                hasattr(m, 'bias') and \
                m.bias is not None:
                    m.bias.requires_grad = True
    else:
        raise NotImplementedError


def lora_state_dict(model: nn.Module, bias: str = 'none') -> Dict[str, torch.Tensor]:
    """Return state_dict with weights of LoRA's A and B matrices and with biases depending on the `bias` value.

    Args:
        model: model with LoRA layers
        bias: 
            ``"none"``: state dict will not store bias weights,
            ``"lora_only"``: state dict will store bias weights only from LoRA layers,
            ``"all"``: state dict will store all bias weights.

    Returns:
        Weights and biases of LoRA layers

    Raises:
        NotImplementedError: if `bias` not in ["none", "lora_only", "all"]
    """
    my_state_dict = model.state_dict()
    if bias == 'none':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k}
    elif bias == 'all':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k or 'bias' in k}
    elif bias == 'lora_only':
        to_return = {}
        for k in my_state_dict:
            if 'lora_' in k:
                to_return[k] = my_state_dict[k]
                bias_name = k.split('lora_')[0]+'bias'
                if bias_name in my_state_dict:
                    to_return[bias_name] = my_state_dict[bias_name]
        return to_return
    else:
        raise NotImplementedError


@dataclass
class LoRAConfig:
    r: float = 0.0
    alpha: float = 1.0
    dropout: float = 0.0

class LoRADINOAttention(DINOAttention):
    lora_config = None
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__(dim, num_heads, qkv_bias, proj_bias, attn_drop, proj_drop)
        if self.lora_config.r > 0:
            self.lora_A_v = nn.Parameter(
                torch.zeros((self.lora_config.r, dim)))  # (4, 128)
            self.lora_B_v = nn.Parameter(
                torch.zeros((dim, self.lora_config.r)))  # (256, 2)
            self.lora_A_q = nn.Parameter(
                torch.zeros((self.lora_config.r, dim)))  # (4, 128)
            self.lora_B_q = nn.Parameter(
                torch.zeros((dim, self.lora_config.r)))  # (256, 2)
            self.lora_scaling = self.lora_config.alpha / self.lora_config.r
            if self.lora_config.dropout > 0.:
                self.lora_dropout = nn.Dropout(p=self.lora_config.dropout)
            else:
                self.lora_dropout = lambda x: x
            self.lora_r = self.lora_config.r
            self.reset_parameters()
        else:
            self.lora_r = 0

    def reset_parameters(self):
        """Reset all the weights, even including pretrained ones."""
        if hasattr(self, 'lora_A_q'):
            # initialize A the same way as the default for nn.Linear and B to zero
            # Wondering why 'a' is equal to math.sqrt(5)?: https://github.com/pytorch/pytorch/issues/15314
            nn.init.kaiming_uniform_(self.lora_A_q, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B_q)
            nn.init.kaiming_uniform_(self.lora_A_v, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B_v)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, C).permute(2, 0, 1, 3)

        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        if self.lora_r > 0:
            previous_dtype = q.dtype
            q = q.to(self.lora_A_q.data.dtype)
            q_after_A = F.linear(self.lora_dropout(x), self.lora_A_q)  # (64, 64, 128) @ (4, 128) -> (64, 64, 4)
            q_after_B = F.linear(q_after_A, self.lora_B_q)
            q = q + q_after_B * self.lora_scaling
            q = q.to(previous_dtype)

            v = v.to(self.lora_A_v.data.dtype)
            v_after_A = F.linear(self.lora_dropout(x), self.lora_A_v)  # (64, 64, 128) @ (4, 128) -> (64, 64, 4)
            v_after_B = F.linear(v_after_A, self.lora_B_v)
            v = v + v_after_B * self.lora_scaling
            v = v.to(previous_dtype)
        q = q.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        k = k.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class LoRALlamaAttention(LlamaAttention):
    lora_config = None

    def __init__(self, config) -> None:
        """Causal self-attention with calculating qkv matrices with a single matrix* and Low Ranking Adaptation for
        paremeter-efficient finetuning.

        *Instead of creating multiple heads and concatenating the result (in addition to creating separate matrices for
        query, key and value for each head) we can do this in a single pass with a single weight matrix.

        Args:
            config: 
                ``"block_size"``: size of the context of the model,
                ``"vocab_size"``: number of unique tokens,
                ``"padded_vocab_size"``: padded size of the vocabulary to the nearest multiple of 64 (leads to a greater performance),
                ``"n_layer"``: number of transformer blocks (self-attention + MLP),
                ``"n_head"``: number of heads in multihead attention mechanism,
                ``"n_embd"``: size of the embedding: vector representation of each token.
        """
        # Skip the parent class __init__ altogether and replace it to avoid
        # useless allocations
        nn.Module.__init__(self)

        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings

        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.pretraining_tp = config.pretraining_tp

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = MergedLinear(
            in_features=self.hidden_size,
            out_features=self.num_heads * self.head_dim,
            r=self.lora_config.r,
            lora_alpha=self.lora_config.alpha,
            lora_dropout=self.lora_config.dropout,
            fan_in_fan_out = False,
            bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = MergedLinear(
            in_features=self.hidden_size,
            out_features=self.num_heads * self.head_dim,
            r=self.lora_config.r,
            lora_alpha=self.lora_config.alpha,
            lora_dropout=self.lora_config.dropout,
            fan_in_fan_out = False,
            bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)


class LoRACLIPAttention(CLIPAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    lora_config = None
    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        # self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        # self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        # self.k_proj = MergedLinear(
        #     in_features=self.embed_dim,
        #     out_features=self.embed_dim,
        #     r=self.lora_config.r,
        #     lora_alpha=self.lora_config.alpha,
        #     lora_dropout=self.lora_config.dropout,
        #     fan_in_fan_out = False,
        #     bias=True)
        
        self.q_proj = MergedLinear(
            in_features=self.embed_dim,
            out_features=self.embed_dim,
            r=self.lora_config.r,
            lora_alpha=self.lora_config.alpha,
            lora_dropout=self.lora_config.dropout,
            fan_in_fan_out = False,
            bias=True)

        self.v_proj = MergedLinear(
            in_features=self.embed_dim,
            out_features=self.embed_dim,
            r=self.lora_config.r,
            lora_alpha=self.lora_config.alpha,
            lora_dropout=self.lora_config.dropout,
            fan_in_fan_out = False,
            bias=True)

        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
        # self.out_proj = MergedLinear(
        #     in_features=self.embed_dim,
        #     out_features=self.embed_dim,
        #     r=self.lora_config.r,
        #     lora_alpha=self.lora_config.alpha,
        #     lora_dropout=self.lora_config.dropout,
        #     fan_in_fan_out = False,
        #     bias=True)


@contextmanager
def lora_dino(r, alpha, dropout, enabled: bool = True):
    """Apply context manager under which you can instantiate the model with LoRA.

    In a nutshell the code inside this function forces to use LoRA variant of causal self-attention
    instead of the original one (without LoRA).

    Args:
        r: rank of the weight update matrices. To make sense of using LoRA the rank should be smaller than the rank of
            the weights of the model.  The rank can be as low as 1: https://arxiv.org/pdf/2106.09685.pdf (section 7.2)
        alpha: alpha is needed for scaling updates as alpha/r
            "This scaling helps to reduce the need to retune hyperparameters when we vary r"
            https://arxiv.org/pdf/2106.09685.pdf (section 4.1)
        dropout: dropout that is applied on the input in the LoRA branch (before multiplying by matrix A)
        enabled: enables/disables LoRA
    """
    if not enabled:
        yield
        return

    LoRADINOAttention.lora_config = LoRAConfig(r=r, alpha=alpha, dropout=dropout)
    # when entering context manager replace link to causal self-attention class from original
    # to a variant with LoRA
    original_self_attention = DINOAttention
    models.dino.MemEffAttention = LoRADINOAttention
    yield
    # when exiting context manager - restore link to original causal self-attention class
    models.dino.MemEffAttention = original_self_attention

    LoRADINOAttention.lora_config = None

@contextmanager
def lora(r, alpha, dropout, enabled: bool = True):
    """Apply context manager under which you can instantiate the model with LoRA.

    In a nutshell the code inside this function forces to use LoRA variant of causal self-attention
    instead of the original one (without LoRA).

    Args:
        r: rank of the weight update matrices. To make sense of using LoRA the rank should be smaller than the rank of
            the weights of the model.  The rank can be as low as 1: https://arxiv.org/pdf/2106.09685.pdf (section 7.2)
        alpha: alpha is needed for scaling updates as alpha/r
            "This scaling helps to reduce the need to retune hyperparameters when we vary r"
            https://arxiv.org/pdf/2106.09685.pdf (section 4.1)
        dropout: dropout that is applied on the input in the LoRA branch (before multiplying by matrix A)
        enabled: enables/disables LoRA
    """
    if not enabled:
        yield
        return

    LoRALlamaAttention.lora_config = LoRAConfig(r=r, alpha=alpha, dropout=dropout)
    # when entering context manager replace link to causal self-attention class from original
    # to a variant with LoRA
    causal_self_attention = LlamaAttention
    transformers.models.llama.modeling_llama.LlamaAttention = LoRALlamaAttention
    yield
    # when exiting context manager - restore link to original causal self-attention class
    transformers.models.llama.modeling_llama.LlamaAttention = causal_self_attention

    LoRALlamaAttention.lora_config = None


@contextmanager
def lora_clip(r, alpha, dropout, enabled: bool = True):
    """Apply context manager under which you can instantiate the model with LoRA.

    In a nutshell the code inside this function forces to use LoRA variant of causal self-attention
    instead of the original one (without LoRA).

    Args:
        r: rank of the weight update matrices. To make sense of using LoRA the rank should be smaller than the rank of
            the weights of the model.  The rank can be as low as 1: https://arxiv.org/pdf/2106.09685.pdf (section 7.2)
        alpha: alpha is needed for scaling updates as alpha/r
            "This scaling helps to reduce the need to retune hyperparameters when we vary r"
            https://arxiv.org/pdf/2106.09685.pdf (section 4.1)
        dropout: dropout that is applied on the input in the LoRA branch (before multiplying by matrix A)
        enabled: enables/disables LoRA
    """
    if not enabled:
        yield
        return

    LoRACLIPAttention.lora_config = LoRAConfig(r=r, alpha=alpha, dropout=dropout)
    # when entering context manager replace link to causal self-attention class from original
    # to a variant with LoRA
    original_self_attention = CLIPAttention
    transformers.models.clip.modeling_clip.CLIPAttention = LoRACLIPAttention
    yield
    # when exiting context manager - restore link to original causal self-attention class
    transformers.models.clip.modeling_clip.CLIPAttention = original_self_attention

    LoRACLIPAttention.lora_config = None