import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence
import random

import torch

import transformers

from utils.llavasimple_trainer import LLaVASimpleTrainer
from models import LocLLMModel
from datasets import Human36MDataset, MPII3DDataset, MemoryEfficientConcatDataset

from PIL import Image
import torch.nn as nn
import io

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
# FIXME: seems wrong?
# DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    llama_path: Optional[str] = field(default="")
    dino_path: Optional[str] = field(default=None)
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    tune_mm_mlp_adapter: bool = field(default=True)
    freeze_vit: bool = field(default=True)
    freeze_llm: bool = field(default=True)


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    dataset_name: str = field(default="")
    image_token_len: int = 0
    image_folder: Optional[str] = field(default=None)
    image_size: int = field(default=224)
    crop_size: int = field(default=224)
    data_augmentation: bool = field(default=False)
    conv_format: str = field(default="keypoint")

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    force_fsdp: bool = field(default=False)
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )

@dataclass
class LoRAArguments:
    lora_vision_r: int = field(default=8)
    lora_vision_alpha: float = field(default=16)
    lora_vision_dropout: float = field(default=0.05)
    lora_vision_enable: bool = field(default=False)
    lora_llm_r: int = field(default=8)
    lora_llm_alpha: float = field(default=16)
    lora_llm_dropout: float = field(default=0.05)
    lora_llm_enable: bool = field(default=False)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            assert all(x is not None and x.shape == images[0].shape for x in images)
            batch['images'] = torch.stack(images)

        assert 'has_image' in instances[0].keys()
        has_images = [instance['has_image'] for instance in instances]
        batch['has_images'] = has_images

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""

    assert "@" in data_args.dataset_name
    # NOTE: use "@" merge dataset without considering the number of samples in the dataset
    datasets = data_args.dataset_name.split("@")
    image_folders = data_args.image_folder.split("@")
    data_paths = data_args.data_path.split("@")
    merge_datasets = []
    repeat_datasets = []
    for data_path, image_folder, dataset in zip(data_paths, image_folders, datasets):
        if ":" in dataset:
            # NOTE: use : for repeat dataset
            dataset, repeat_number = dataset.split(":")
            repeat_datasets.append(int(repeat_number))
        else:
            repeat_datasets.append(1)
        dataset_cls = eval(dataset)
        train_dataset = dataset_cls(tokenizer=tokenizer,
                                    data_path=data_path,
                                    multimodal_cfg=dict(
                                    image_folder=image_folder,
                                    data_augmentation=data_args.data_augmentation,
                                    image_size=data_args.image_size,
                                    crop_size=data_args.crop_size,
                                    conv_format=data_args.conv_format))
        merge_datasets.append(train_dataset)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=MemoryEfficientConcatDataset(merge_datasets, repeats=repeat_datasets),
                eval_dataset=None,
                data_collator=data_collator)
    
def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoRAArguments))
    model_args, data_args, training_args, lora_args = parser.parse_args_into_dataclasses()

    model = LocLLMModel.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        llama_path=model_args.llama_path,
        dino_path=model_args.dino_path,
        lora_vision_r=lora_args.lora_vision_r,
        lora_vision_alpha=lora_args.lora_vision_alpha,
        lora_vision_dropout=lora_args.lora_vision_dropout,
        lora_vision_enable=lora_args.lora_vision_enable,
        lora_llm_enable=lora_args.lora_llm_enable,
        lora_llm_r=lora_args.lora_llm_r,
        lora_llm_alpha=lora_args.lora_llm_alpha,
        lora_llm_dropout=lora_args.lora_llm_dropout,
        crop_size=data_args.crop_size)
    
    # load mm projector weights
    if model_args.pretrain_mm_mlp_adapter is not None:
        print('Load pretrained mm_projector from: ', model_args.pretrain_mm_mlp_adapter)
        mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
        update_state = {}
        update_state['weight'] = mm_projector_weights['model.mm_projector.weight']
        update_state['bias'] = mm_projector_weights['model.mm_projector.bias']
        model.mm_projector.load_state_dict(update_state, strict=True)

    model.config.use_cache = False

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    tokenizer.pad_token = tokenizer.unk_token

    model.initialize_vision_tokenizer(tokenizer=tokenizer)

    dtype = torch.bfloat16
    model.model.to(dtype)
    model.lm_head.to(dtype)

    for param in model.parameters():
        param.requires_grad_(False)

    if model_args.tune_mm_mlp_adapter:
        for p in model.mm_projector.parameters():
            p.requires_grad = True

    data_args.image_token_len = model.config.num_patches

    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)

    if not model_args.freeze_vit:
        assert model.config.lora_vision_enable
        for name, param in model.vision_model.named_parameters():
            if "lora_" not in name:
                param.requires_grad = False
            else:
                param.data = param.data.float()
                param.requires_grad = True
    else:
        model.vision_model.train = disabled_train
        model.vision_model.eval()

    if not model_args.freeze_llm:
        assert model.config.lora_llm_enable
        for name, param in model.model.named_parameters():
            if "lora_" not in name:
                param.requires_grad = False
            else:
                param.data = param.data.float()
                param.requires_grad = True

    params_grad = [n for n, p in model.named_parameters() if p.requires_grad]
    print("param_grad: {}".format(params_grad))
    # NOTE: enable grad on embedding for gradient checkpoint
    # for p in model.get_input_embeddings().parameters():
    #     p.requires_grad = True
    trainer = LLaVASimpleTrainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    **data_module)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer,
                                   output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
