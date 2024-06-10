import argparse
from transformers import AutoTokenizer, AutoConfig
import torch
import os
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F
import pickle as pk
import time

import math
from models import LocLLMModel
from datasets.h36m import H36M_KEYPOINT_NAME, Human36MDataset, KeypointLocationDescription, KeypointLocationQuestion, transform_preds, pixel2cam, reconstruction_error
from datasets.convsersation import conv_keypoint, conv_llama2, conv_simple
from dataclasses import dataclass
import re

def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
PREFIX_IMAGE = "Image: "
PREFIX_NO_IMAGE = "Image: N/A"
BEGIN_DESCRIPTION = "<des>"
END_DESCRIPTION = "</des>"

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
            for kpt_id, kpt_name in enumerate(H36M_KEYPOINT_NAME):
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

    kpt_pred = {}
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
            output_ids = model.generate(
                input_ids,
                images=batch_images,
                has_images=batch_has_images,
                attention_mask=attention_mask,
                do_sample=False,
                max_new_tokens=22
            )

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

        pattern = re.compile(r'\.\d+')

        for i in range(len(outputs)):
            # decode coordinates from token
            pred_kpt = outputs[i]
            res = pattern.findall(pred_kpt)
            if not len(res) == 3: 
                print('Format error', pred_kpt)
            if len(res) == 0: continue
            x, y, z = float('0'+res[0]), float('0'+res[1]), float('0'+res[2])
            x = x * crop_size
            y = y * crop_size
            z = (z - 0.5)

            decoded_kpt[i, 0] = x
            decoded_kpt[i, 1] = y
            decoded_kpt[i, 2] = z

        decoded_kpt[0, 2] = 0.0

        decoded_kpt[:, :2] = transform_preds(decoded_kpt[:, :2], c, s, [crop_size, crop_size])

        kpt_pred[int(image_id)] = decoded_kpt
    
    with open(os.path.join(output_dir, f'test_gt_kpt_rank_{rank}.pkl'), 'wb') as fid:
        pk.dump(kpt_pred, fid, pk.HIGHEST_PROTOCOL)

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
        kpt_all_pred = {}
        for r in range(world_size):
            with open(os.path.join(output_dir, f'test_gt_kpt_rank_{r}.pkl'), 'rb') as fid:
                kpt_pred = pk.load(fid)

            # os.remove(os.path.join(output_dir, f'test_gt_kpt_rank_{r}.pkl'))

            kpt_all_pred.update(kpt_pred)

        res_file = os.path.join(args.output_dir, 'pred_kpt.json')
        with open(res_file, 'wb') as fid:
            pk.dump(kpt_all_pred, fid, pk.HIGHEST_PROTOCOL)

        preds = kpt_all_pred

        gts = dataset.list_data_dict

        assert len(gts) == len(preds)
        sample_num = len(gts)
        num_joints = 17
        pred_save = []
        error = np.zeros((sample_num, num_joints))  # joint error
        pa_error = np.zeros((sample_num, num_joints))  # joint error
        error_x = np.zeros((sample_num, num_joints))  # joint error
        error_y = np.zeros((sample_num, num_joints))  # joint error
        error_z = np.zeros((sample_num, num_joints))  # joint error
        # error for each sequence
        action_name = ['Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning', 'Posing', 'Purchases',
                    'Sitting', 'SittingDown', 'Smoking', 'Photo', 'Waiting', 'Walking', 'WalkDog', 'WalkTogether']
        error_action = [[] for _ in range(len(action_name))]
        EVAL_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        root_idx = 0
        for n in range(sample_num):
            gt = gts[n]
            image_id = gt['img_id']
            f = gt['f']
            c = gt['c']
            bbox = gt['bbox']
            gt_3d_root = gt['root_cam']
            gt_3d_kpt = gt['joint_cam']

            gt_3d_kpt = np.take(gt_3d_kpt, EVAL_JOINTS, axis=0)
            # gt_vis = gt['joint_vis']

            # restore coordinates to original space
            pred_2d_kpt = preds[image_id].copy()
            # pred_2d_kpt[:, 0] = pred_2d_kpt[:, 0] / self._output_size[1] * bbox[2] + bbox[0]
            # pred_2d_kpt[:, 1] = pred_2d_kpt[:, 1] / self._output_size[0] * bbox[3] + bbox[1]
            pred_2d_kpt[:, 2] = pred_2d_kpt[:, 2] * 2000 + gt_3d_root[2]

            # back project to camera coordinate system
            pred_3d_kpt = pixel2cam(pred_2d_kpt, f, c)

            # root joint alignment
            pred_3d_kpt = pred_3d_kpt - pred_3d_kpt[root_idx]
            gt_3d_kpt = gt_3d_kpt - gt_3d_kpt[root_idx]

            # exclude thorax
            pred_3d_kpt = np.take(pred_3d_kpt, EVAL_JOINTS, axis=0)
            gt_3d_kpt = np.take(gt_3d_kpt, EVAL_JOINTS, axis=0)

            # rigid alignment for PA MPJPE (protocol #1)
            aligned_pred_3d_kpt = reconstruction_error(pred_3d_kpt.copy(), gt_3d_kpt)

            # error calculate
            error[n] = np.sqrt(np.sum((pred_3d_kpt - gt_3d_kpt)**2, 1))
            pa_error[n] = np.sqrt(np.sum((aligned_pred_3d_kpt - gt_3d_kpt)**2, 1))
            error_x[n] = np.abs(pred_3d_kpt[:, 0] - gt_3d_kpt[:, 0])
            error_y[n] = np.abs(pred_3d_kpt[:, 1] - gt_3d_kpt[:, 1])
            error_z[n] = np.abs(pred_3d_kpt[:, 2] - gt_3d_kpt[:, 2])
            img_name = gt['file_name']
            action_idx = int(img_name[img_name.find(
                'act') + 4:img_name.find('act') + 6]) - 2
            error_action[action_idx].append(error[n].copy())

            # prediction save
            pred_save.append({'image_id': image_id, 'joint_cam': pred_3d_kpt.tolist(
            ), 'bbox': bbox, 'root_cam': gt_3d_root.tolist()})  # joint_cam is root-relative coordinate

        # total error
        tot_err = np.mean(error)
        tot_pa_error = np.mean(pa_error)
        tot_err_x = np.mean(error_x)
        tot_err_y = np.mean(error_y)
        tot_err_z = np.mean(error_z)
        metric = 'MPJPE'

        eval_summary = f'Protocol {2} error ({metric}) >> tot: {tot_err:2f}, x: {tot_err_x:2f}, y: {tot_err_y:.2f}, z: {tot_err_z:2f}\n PA MPJPE: {tot_pa_error}\n'

        # error for each action
        for i in range(len(error_action)):
            err = np.mean(np.array(error_action[i]))
            eval_summary += (action_name[i] + ': %.2f ' % err)

        print(eval_summary)

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

    dataset = Human36MDataset(tokenizer=None,
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
