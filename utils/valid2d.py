import argparse
from transformers import AutoTokenizer, AutoConfig
import torch
import os
import json
from tqdm import tqdm
from transformers import StoppingCriteria
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F
import pickle as pk
import time

import math
from models import LocLLMModel
from datasets.coco import COCODataset, COCO_KEYPOINT_NAME, KeypointLocationDescription, KeypointLocationQuestion, transform_preds
from datasets.convsersation import conv_keypoint, conv_llama2, conv_simple
from dataclasses import dataclass
import re
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
PREFIX_IMAGE = "Image: "

@dataclass
class DataCollatorForSupervisedDataset(object):
    def __init__(self, image_token_len, conv_format):
        self.image_token_len = image_token_len
        self.conv_format = conv_format

    def __call__(self, instances):
        """Collate examples for supervised fine-tuning."""
        batch_prompts = []
        batch_images = []
        batch_has_images = []
        result_dicts = []

        if self.conv_format == 'simple':
            conv = conv_simple.copy()
        elif self.conv_format == 'keypoint':
            conv = conv_keypoint.copy()
        else:
            conv = conv_llama2.copy()

        for i, line in enumerate(instances):
            result_dict = {}
            images = line['images'].unsqueeze(0)
            image_id = line['image_id']
            c = line['c']
            s = line['s']
            for kpt_id, kpt_name in enumerate(COCO_KEYPOINT_NAME):
                question = KeypointLocationQuestion[kpt_name][0]
                kpt_des = KeypointLocationDescription[kpt_name]

                conv.messages = []
                if self.conv_format == 'keypoint':
                    q1 = "Where is the {} of this person in this image? Please provide its coordinates.".format(kpt_name)
                    conv.append_message(conv.roles[0], kpt_des)
                    conv.append_message(conv.roles[1], q1)
                    conv.append_message(conv.roles[2], None)
                elif self.conv_format == 'simple':
                    q1 = "Where is the {} of this person in this image? Please provide its coordinates.".format(kpt_name)
                    conv.append_message(conv.roles[0], q1)
                    conv.append_message(conv.roles[1], None)
                else:
                    conv.append_message(conv.roles[0], question)
                    conv.append_message(conv.roles[1], None)
                
                if self.conv_format == 'llama2':
                    conv.system = "[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n".format(system_message=PREFIX_IMAGE + self.image_token_len * DEFAULT_IMAGE_PATCH_TOKEN)
                    cur_prompt = conv.get_prompt()
                else:
                    text_inputs = conv.get_prompt()
                    cur_prompt = PREFIX_IMAGE + self.image_token_len * DEFAULT_IMAGE_PATCH_TOKEN + "\n" + text_inputs

                has_images = True

                result_dict['initial_prompt'] = cur_prompt
                result_dict['image_id'] = image_id
                result_dict['c'] = c
                result_dict['s'] = s
                batch_prompts.append(cur_prompt)
                batch_images.append(images)
                batch_has_images.append(has_images)
                result_dicts.append(result_dict)

        return result_dicts, batch_prompts, batch_images, batch_has_images



@torch.no_grad()
def worker(model, tokenizer, dataset, args, output_dir):
    crop_size = model.config.crop_size
    image_token_len = model.config.num_patches

    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    indices = list(range(rank, len(dataset), world_size))
    print("==>" + " Worker {} Started, responsible for {} images".format(rank, len(indices)))

    sub_dataset = torch.utils.data.Subset(dataset, indices)
    batch_size = 1
    data_loader = DataLoader(sub_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=DataCollatorForSupervisedDataset(image_token_len, args.conv_format))

    all_preds = []
    for result_dicts, batch_prompts, batch_images, batch_has_images in tqdm(data_loader):
        assert len(result_dicts) == 17
        # inputs = tokenizer()
        tokenized_output = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        batch_images = torch.cat(batch_images, dim=0).cuda()
        assert batch_images.shape[0] == 17

        input_ids = torch.as_tensor(tokenized_output.input_ids).cuda()
        attention_mask = torch.as_tensor(tokenized_output.attention_mask).cuda()

        with torch.inference_mode():
            output_dict = model.generate(
                input_ids,
                images=batch_images,
                has_images=batch_has_images,
                attention_mask=attention_mask,
                do_sample=False,
                max_new_tokens=13,
                output_scores=True,
                return_dict_in_generate=True
            )
            output_ids = output_dict['sequences']
            output_scores = output_dict['scores']

        outputs = []
        for input_id, output_id in zip(input_ids, output_ids):
            input_token_len = input_id.shape[0]
            n_diff_input_output = (input_id != output_id[:input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] Sample {i}: {n_diff_input_output} output_ids are not the same as the input_ids')
            output = tokenizer.batch_decode(output_id[input_token_len:].unsqueeze(0), skip_special_tokens=True)[0]
            output = output.strip()
            outputs.append(output)

        assert len(outputs) == 17
        decoded_kpt = np.zeros((17, 3))
        image_id = result_dicts[0]['image_id']
        c = result_dicts[0]['c']
        s = result_dicts[0]['s']

        output_scores = torch.stack(output_scores[:-1], dim=0)
        pattern = re.compile(r'0\.\d+')

        for i in range(len(outputs)):
            # decode coordinates from token
            pred_kpt = outputs[i]
            res = pattern.findall(pred_kpt)
            if not len(res) == 2: 
                print('Format error', pred_kpt)
            if len(res) == 0: continue
            if len(res) == 1:
                x = float(res[0]) * crop_size
                x_pos = pred_kpt.find(res[0])
                x_s = output_scores[x_pos:x_pos+len(res[0]), i, :].cpu()
                x_s = F.softmax(x_s, dim=1)
                x_s = torch.max(x_s, dim=1)[0].mean().float().item()
                y = 0
                y_s = 0
            else:
                x, y = float(res[0]), float(res[1])
                x, y = x * crop_size, y * crop_size
            
                x_pos = pred_kpt.find(res[0])
                x_s = output_scores[x_pos:x_pos+len(res[0]), i, :].cpu()
                x_s = F.softmax(x_s, dim=1)
                x_s = torch.max(x_s, dim=1)[0].mean().float().item()
                y_pos = pred_kpt.find(res[1])
                y_s = output_scores[y_pos:y_pos+len(res[1]), i, :].cpu()
                y_s = F.softmax(y_s, dim=1)
                y_s = torch.max(y_s, dim=1)[0].mean().float().item()

            decoded_kpt[i, 0] = x
            decoded_kpt[i, 1] = y
            decoded_kpt[i, 2] = (x_s + y_s) / 2.0

        decoded_kpt[:, :2] = transform_preds(
            decoded_kpt[:, :2], c, s, (crop_size, crop_size)
        )

        data = dict()
        data['image_id'] = image_id
        data['score'] = float(np.mean(decoded_kpt[:, 2]))
        data['keypoints'] = decoded_kpt.reshape(-1).tolist()
        data['category_id'] = 1
        
        all_preds.append(data)
    
    with open(os.path.join(output_dir, f'test_gt_kpt_rank_{rank}.pkl'), 'wb') as fid:
        pk.dump(all_preds, fid, pk.HIGHEST_PROTOCOL)

    torch.distributed.barrier()  # Make sure all JSON files are saved

    if rank == 0:
        # manually sleep to wait all file are saved
        while True:
            ready = True
            for r in range(world_size):
                if not os.path.exists(os.path.join(output_dir, f'test_gt_kpt_rank_{r}.pkl')):
                    ready = False
            if ready: 
                break
            else:
                time.sleep(20)
        # sleep 30s to make sure all files are saved
        time.sleep(20)
        kpt_all_pred = []
        for r in range(world_size):
            with open(os.path.join(output_dir, f'test_gt_kpt_rank_{r}.pkl'), 'rb') as fid:
                kpt_pred = pk.load(fid)

            # os.remove(os.path.join(output_dir, f'test_gt_kpt_rank_{r}.pkl'))

            kpt_all_pred.extend(kpt_pred)

        ann_file = args.question_file
        res_file = os.path.join(output_dir, 'pred_kpt.json')
        with open(res_file, 'w') as fid:
            json.dump(kpt_all_pred, fid)

        cocoGt = COCO(ann_file)
        cocoDt = cocoGt.loadRes(res_file)

        cocoEval = COCOeval(cocoGt, cocoDt, 'keypoints')
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        return True
    else:
        return False

def eval_model(args):
    torch.distributed.init_process_group(backend='nccl')
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    print('Init process group: world_size: {}, rank: {}'.format(world_size, rank))
    torch.cuda.set_device(rank)

    disable_torch_init()
    model_name = os.path.expanduser(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, padding_side='left')

    model = LocLLMModel.from_pretrained(model_name, use_cache=True)
    for name, param in model.model.named_parameters():
        if "lora_" not in name:
            param.data = param.data.bfloat16()
    model.lm_head.to(torch.bfloat16)
    model = model.cuda()
    # model = torch.nn.parallel.DistributedDataParallel(model)

    dataset = COCODataset(tokenizer=None,
                        data_path=os.path.join(args.question_file),
                        multimodal_cfg=dict(
                            image_folder=args.image_folder,
                            image_size=224,
                            crop_size=224,
                            conv_format=args.conv_format),
                            is_train=False)

    worker(model, tokenizer, dataset, args, args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="facebook/opt-350m")
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.json")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-format", type=str, default="keypoint")
    parser.add_argument("--output-dir", type=str, default="")
    args = parser.parse_args()

    eval_model(args)
