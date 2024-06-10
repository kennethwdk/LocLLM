import os
import torch
import torch.nn as nn

from transformers import Trainer
from typing import Dict, Optional, Sequence


def unwrap_model(model: nn.Module) -> nn.Module:
    """
    Recursively unwraps a model from potential containers (as used in distributed training).

    Args:
        model (`torch.nn.Module`): The model to unwrap.
    """
    # since there could be multiple levels of wrapping, unwrap recursively
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    else:
        return model


class LLaVASimpleTrainer(Trainer):
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        model_to_save = unwrap_model(self.model)
        # if state_dict is None:
            # state_dict = model_to_save.state_dict()
        if hasattr(model_to_save, "orig_embeds_params"):
            assert model_to_save.original_tokens_length == model_to_save.orig_embeds_params[0].shape[0]
            model_to_save.get_input_embeddings().weight.data[:model_to_save.original_tokens_length].copy_(model_to_save.orig_embeds_params[0].clone().detach())
            assert model_to_save.orig_lm_head is not None
            model_to_save.get_output_embeddings().weight.data[:model_to_save.original_tokens_length].copy_(model_to_save.orig_lm_head[0].clone().detach())
            print("back parameters")
            # state_dict['model.embed_tokens.weight'][:model_to_save.original_tokens_length] = model_to_save.orig_embeds_params[0].clone().to(device=state_dict['model.embed_tokens.weight'].device, dtype=state_dict['model.embed_tokens.weight'].dtype)
            # state_dict['lm_head.weight'][:model_to_save.original_tokens_length] = model_to_save.orig_embeds_params[0].clone().to(device=state_dict['model.embed_tokens.weight'].device, dtype=state_dict['model.embed_tokens.weight'].dtype)
        state_dict = model_to_save.state_dict()

        super(LLaVASimpleTrainer, self)._save(output_dir, state_dict)
