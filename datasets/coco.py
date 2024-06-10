import transformers
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import logging
import random
from typing import Dict
import os
import numpy as np
from pycocotools.coco import COCO
import cv2

DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
PREFIX_IMAGE = "Image: "
PREFIX_NO_IMAGE = "Image: N/A"
BEGIN_DESCRIPTION = "<des>"
END_DESCRIPTION = "</des>"
IGNORE_INDEX = -100
DEFAULT_EOS_TOKEN = "</s>"

from .constants import COCO_KEYPOINT_NAME, KeypointLocationDescription, KeypointLocationQuestion
from .convsersation import conv_simple, conv_llama2, conv_keypoint

DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
PREFIX_IMAGE = "Image: "
PREFIX_NO_IMAGE = "Image: N/A"
BEGIN_DESCRIPTION = "<des>"
END_DESCRIPTION = "</des>"
IGNORE_INDEX = -100
DEFAULT_EOS_TOKEN = "</s>"
BEGIN_OPTIONS = "<opt>"
END_OPTIONS = "</opt>"
BEGIN_LOC = "<loc>"
END_LOC = "</loc>"
BEGIN_QUESTION = "<qes>"
END_QUESTION = "</qes>"

class COCODataset(Dataset):
    """Dataset for supervised fine-tuning."""
    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 multimodal_cfg: dict,
                 is_train=True
                 ):
        super(COCODataset, self).__init__()
        logging.warning("Loading data...")
        self.size = 224
        self.aspect_ratio = 1.0
        self.pixel_std = 200
        self.num_joints = 17
        coco = COCO(data_path)
        list_data_dict = []
        instance_id = 0
        for index in coco.getImgIds():
            im_ann = coco.loadImgs(index)[0]
            width = im_ann['width']
            height = im_ann['height']
            annIds = coco.getAnnIds(imgIds=index, iscrowd=False)
            objs = coco.loadAnns(annIds)
            # sanitize bboxes
            valid_objs = []
            for obj in objs:
                x, y, w, h = obj['bbox']
                x1 = np.max((0, x))
                y1 = np.max((0, y))
                x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
                y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
                if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
                    obj['clean_bbox'] = [x1, y1, x2-x1, y2-y1]
                    valid_objs.append(obj)
            objs = valid_objs

            for obj in objs:
                cls = obj['category_id']
                if cls != 1: continue

                # ignore objs without keypoints annotation
                if max(obj['keypoints']) == 0:
                    continue

                joints_3d = np.zeros((self.num_joints, 3), dtype=np.float32)
                joints_3d_vis = np.zeros((self.num_joints, 3), dtype=np.float32)
                visible = np.zeros((self.num_joints), dtype=np.float32)
                for ipt in range(self.num_joints):
                    joints_3d[ipt, 0] = obj['keypoints'][ipt * 3 + 0]
                    joints_3d[ipt, 1] = obj['keypoints'][ipt * 3 + 1]
                    joints_3d[ipt, 2] = 0
                    t_vis = obj['keypoints'][ipt * 3 + 2]
                    visible[ipt] = t_vis
                    if t_vis > 1:
                        t_vis = 1
                    joints_3d_vis[ipt, 0] = t_vis
                    joints_3d_vis[ipt, 1] = t_vis
                    joints_3d_vis[ipt, 2] = 0

                center, scale = self._box2cs(obj['clean_bbox'][:4])
                list_data_dict.append({
                    'file_name': im_ann['file_name'],
                    'image_id': index,
                    'center': center,
                    'scale': scale,
                    'joints_3d': joints_3d,
                    'joints_3d_vis': joints_3d_vis,
                    'instance_id': instance_id
                })
                instance_id += 1
        
        logging.warning("The number of training samples is {}".format(len(list_data_dict)))
        logging.warning("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.multimodal_cfg = multimodal_cfg
        self.conv_format = self.multimodal_cfg.get("conv_format", "keypoint")
        if self.conv_format == 'simple':
            self.conv = conv_simple.copy()
        elif self.conv_format == 'keypoint':
            self.conv = conv_keypoint.copy()
        else:
            self.conv = conv_llama2.copy()
        print('Use Conv Format ', self.conv_format)
        if 'data_augmentation' in self.multimodal_cfg.keys():
            self.data_aug = self.multimodal_cfg['data_augmentation']
        else:
            self.data_aug = False
        if self.multimodal_cfg.get('dino_norm', False):
            norm_mean = (0.485, 0.456, 0.406)
            norm_std = (0.229, 0.224, 0.225)
        else:
            norm_mean = (0.48145466, 0.4578275, 0.40821073)
            norm_std = (0.26862954, 0.26130258, 0.27577711)
        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(norm_mean, norm_std),
            ]
        )
        self.is_train = is_train

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i):
        if self.is_train:
            while True:
                use_item, data_dict = self._parse_data_item(i)
                if use_item:
                    break
                else:
                    i = random.randint(0, self.__len__() - 1)
            return data_dict
        else:
            return self._parse_data_item_val(i)

    def _parse_data_item_val(self, i):
        sources = self.list_data_dict[i]
        result_dict = {}
        image, joints, joints_vis, c, s = self._get_pose_item(sources)
        image_id = sources['image_id']
        result_dict['images'] = image
        result_dict['image_id'] = image_id
        result_dict['c'] = c
        result_dict['s'] = s
        result_dict['joints'] = joints
        result_dict['joints_vis'] = joints_vis
        return result_dict
    
    def _parse_data_item(self, i) -> Dict[str, torch.Tensor]:
        use_item = False
        sources = self.list_data_dict[i]
        data_dict = {}
        
        image, joints, joints_vis, _, _ = self._get_pose_item(sources)

        data_dict["image"] = image
        data_dict["has_image"] = True
        cur_token_len = 256

        # random choice one keypoint for training
        kpt_ids = list(range(self.num_joints))
        np.random.shuffle(kpt_ids)
        is_select = False
        kpt_name = []
        kpt_des = []
        question = []
        caption = []
        for idx in kpt_ids:
            # if idx not in [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 14, 15]: continue
            x, y, v = joints[idx, 0], joints[idx, 1], joints_vis[idx, 0]
            if v < 1: continue
            if x < 0 or x >= self.size or y < 0 or y >= self.size: continue
            x = x / self.size
            y = y /self.size
            location_tokens = "[{:.3f},{:.3f}]".format(x, y)
            kpt_name.append(COCO_KEYPOINT_NAME[idx])
            question.append(KeypointLocationQuestion[COCO_KEYPOINT_NAME[idx]][0])
            kpt_des.append(KeypointLocationDescription[COCO_KEYPOINT_NAME[idx]])
            caption.append(location_tokens)
            is_select = True
            
        if not is_select:
            return use_item, {}
        
        self.conv.messages = []
        for idx in range(5):
            if idx >= len(kpt_des): break
            if self.conv_format == 'keypoint':
                q1 = "Where is the {} of this person in this image? Please provide its coordinates.".format(kpt_name[idx])
                self.conv.append_message(self.conv.roles[0], kpt_des[idx])
                self.conv.append_message(self.conv.roles[1], q1)
                self.conv.append_message(self.conv.roles[2], caption[idx])
            elif self.conv_format == 'simple':
                q1 = "Where is the {} of this person in this image? Please provide its coordinates.".format(kpt_name[idx])
                self.conv.append_message(self.conv.roles[0], q1)
                self.conv.append_message(self.conv.roles[1], caption[idx])
            else:
                self.conv.append_message(self.conv.roles[0], question[idx])
                self.conv.append_message(self.conv.roles[1], caption[idx])

        
        if self.conv_format == 'llama2':
            self.conv.system = "[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n".format(system_message=PREFIX_IMAGE + cur_token_len * DEFAULT_IMAGE_PATCH_TOKEN)
            text_inputs = self.conv.get_prompt()
        else:
            text_inputs = self.conv.get_prompt()
            text_inputs = PREFIX_IMAGE + cur_token_len * DEFAULT_IMAGE_PATCH_TOKEN + "\n" + text_inputs
            

        inputs = self.tokenizer(text_inputs,
                return_tensors="pt",
                padding="longest",
                max_length=self.tokenizer.model_max_length,
                truncation=True).input_ids[0]
        
        if self.conv_format == 'keypoint':
            target = inputs.clone()
            sep = self.conv.sep1 + self.conv.roles[2] + ": "
            rounds = text_inputs.split(self.conv.sep2)
            cur_len = 1
            target[:cur_len] = IGNORE_INDEX
            for i, rou in enumerate(rounds):
                if rou == "":
                    break
                parts = rou.split(sep)
                if len(parts) != 2:
                    break
                parts[0] += sep
                round_len = len(self.tokenizer(rou).input_ids)
                instruction_len = len(self.tokenizer(parts[0]).input_ids) - 2   # <s> <space>
                target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
                cur_len += round_len
        elif self.conv_format == 'llama2':
            target = inputs.clone()
            sep = self.conv.sep + self.conv.roles[1] + " "
            rounds = text_inputs.split(self.conv.sep2)
            cur_len = 1
            target[:cur_len] = IGNORE_INDEX
            for i, rou in enumerate(rounds):
                if rou == "":
                    break
                parts = rou.split(sep)
                if len(parts) != 2:
                    break
                parts[0] += sep
                round_len = len(self.tokenizer(rou).input_ids) + 2
                instruction_len = len(self.tokenizer(parts[0]).input_ids) - 2   # <s> <space>
                target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
                cur_len += round_len
        else:
            target = inputs.clone()
            sep = self.conv.sep + self.conv.roles[1] + ": "
            rounds = text_inputs.split(self.conv.sep2)
            cur_len = 1
            target[:cur_len] = IGNORE_INDEX
            for i, rou in enumerate(rounds):
                if rou == "":
                    break
                parts = rou.split(sep)
                if len(parts) != 2:
                    break
                parts[0] += sep
                round_len = len(self.tokenizer(rou).input_ids)
                instruction_len = len(self.tokenizer(parts[0]).input_ids) - 2   # <s> <space>
                target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
                cur_len += round_len
        
        data_dict.update(
            dict(input_ids=inputs,
                labels=target)
        )

        return True, data_dict
    
    def _get_pose_item(self, sources):
        file_name = sources['file_name']
        image_folder = self.multimodal_cfg['image_folder']
        image_file = os.path.join(image_folder, file_name)
        image = cv2.imread(
            image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
        )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # process image
        joints = sources['joints_3d']
        joints_vis = sources['joints_3d_vis']
        c = sources['center']
        s = sources['scale']
        r = 0

        if self.data_aug:
            sf = 0.3
            rf = 40
            s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
            r = random.uniform(-rf, rf) if random.random() <= 0.5 else 0

            flip_pairs = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]
            # flip
            if random.random() <= 0.5:
                image = image[:, ::-1, :]
                joints, joints_vis = fliplr_joints(
                    joints, joints_vis, image.shape[1], flip_pairs)
                c[0] = image.shape[1] - c[0] - 1

        trans = get_affine_transform(c, s, r, (int(self.size), int(self.size)))
        image = cv2.warpAffine(
            image,
            trans,
            (int(self.size), int(self.size)),
            flags=cv2.INTER_LINEAR)
        image = self.transforms(image)

        for i in range(self.num_joints):
            if joints_vis[i, 0] > 0.0:
                joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)
        
        return image, joints, joints_vis, c, s
    
    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array(
            [w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std],
            dtype=np.float32)
        if center[0] != -1:
            # scale = scale * 1.25
            scale = scale * 1.0

        return center, scale


def fliplr_joints(joints, joints_vis, width, matched_parts):
    """
    flip coords
    """
    # Flip horizontal
    joints[:, 0] = width - joints[:, 0] - 1

    # Change left-right parts
    for pair in matched_parts:
        joints[pair[0], :], joints[pair[1], :] = \
            joints[pair[1], :], joints[pair[0], :].copy()
        joints_vis[pair[0], :], joints_vis[pair[1], :] = \
            joints_vis[pair[1], :], joints_vis[pair[0], :].copy()

    return joints*joints_vis, joints_vis

def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords

def get_affine_transform(
        center, scale, rot, output_size,
        shift=np.array([0, 0], dtype=np.float32), inv=0
):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        print(scale)
        scale = np.array([scale, scale])

    scale_tmp = scale * 200.0
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result
