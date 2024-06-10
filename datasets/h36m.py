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
import json

DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
PREFIX_IMAGE = "Image: "
PREFIX_NO_IMAGE = "Image: N/A"
BEGIN_DESCRIPTION = "<des>"
END_DESCRIPTION = "</des>"
IGNORE_INDEX = -100
DEFAULT_EOS_TOKEN = "</s>"


from .constants import H36M_KEYPOINT_NAME, KeypointLocationDescription, KeypointLocationQuestion
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

class Human36MDataset(Dataset):
    """Dataset for supervised fine-tuning."""
    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 multimodal_cfg: dict,
                 is_train=True
                 ):
        super(Human36MDataset, self).__init__()
        logging.warning("Loading data...")
        self.aspect_ratio = 1
        self.pixel_std = 200
        self.num_joints = 17
        self.root_idx = 0
        with open(data_path, 'r') as fid:
            database = json.load(fid)
        list_data_dict = []

        block_list = ['s_09_act_05_subact_02_ca', 's_09_act_10_subact_02_ca', 's_09_act_13_subact_01_ca']
        for ann_image, ann_annotations in zip(database['images'], database['annotations']):
            ann = dict()
            for k, v in ann_image.items():
                assert k not in ann.keys()
                ann[k] = v
            for k, v in ann_annotations.items():
                ann[k] = v
            skip = False
            for name in block_list:
                if name in ann['file_name']:
                    skip = True
            if skip:
                continue

            image_id = ann['image_id']

            width, height = ann['width'], ann['height']
            xmin, ymin, xmax, ymax = bbox_clip_xyxy(
                bbox_xywh_to_xyxy(ann['bbox']), width, height)

            R, t = np.array(ann['cam_param']['R'], dtype=np.float32), np.array(
                ann['cam_param']['t'], dtype=np.float32)
            f, c = np.array(ann['cam_param']['f'], dtype=np.float32), np.array(
                ann['cam_param']['c'], dtype=np.float32)

            joint_world = np.array(ann['keypoints_world'])
            joint_cam = np.zeros((self.num_joints, 3))
            for j in range(self.num_joints):
                joint_cam[j] = world2cam(joint_world[j], R, t)

            joint_img = cam2pixel(joint_cam, f, c)
            joint_img[:, 2] = joint_img[:, 2] - joint_cam[self.root_idx, 2]
            joint_vis = np.ones((self.num_joints, 3))

            root_cam = joint_cam[self.root_idx]

            abs_path = ann['file_name']

            list_data_dict.append({
                'bbox': (xmin, ymin, xmax, ymax),
                'img_id': image_id,
                'file_name': abs_path,
                'width': width,
                'height': height,
                'joint_img': joint_img,
                'joint_vis': joint_vis,
                'joint_cam': joint_cam,
                'root_cam': root_cam,
                'f': f,
                'c': c
            })
        
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
        if 'crop_size' in self.multimodal_cfg.keys():
            self.size = self.multimodal_cfg['crop_size']
        else:
            self.size = 224
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
        image, _, _, c, s = self._get_pose_item(sources)
        image_id = sources['img_id']
        result_dict['images'] = image
        result_dict['image_id'] = image_id
        result_dict['c'] = c
        result_dict['s'] = s
        return result_dict
    
    def _parse_data_item(self, i) -> Dict[str, torch.Tensor]:
        use_item = False
        sources = self.list_data_dict[i]
        data_dict = {}
        
        image, joints, joints_vis, _, _ = self._get_pose_item(sources)

        data_dict["image"] = image
        data_dict['has_image'] = True
        cur_token_len = 256

        # random choice one keypoint for training
        kpt_ids = list(range(self.num_joints))
        np.random.shuffle(kpt_ids)
        is_select = False
        kpt_name = []
        kpt_des = []
        caption = []
        for idx in kpt_ids:
            x, y, z, v = joints[idx, 0], joints[idx, 1], joints[idx, 2], joints_vis[idx, 0]
            if v < 1: continue
            if x < 0 or x >= 1 or y < 0 or y >= 1 or z < 0 or z > 1: continue
            location_tokens = "[{:.3f},{:.3f},{:.6f}]".format(x, y, z)
            kpt_name.append(H36M_KEYPOINT_NAME[idx])
            kpt_des.append(KeypointLocationDescription[H36M_KEYPOINT_NAME[idx]])
            caption.append(location_tokens)
            is_select = True
            
        if not is_select:
            return use_item, {}
        
        self.conv.messages = []
        for idx in range(4):
            if idx >= len(kpt_name): continue
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
        
        joints = sources['joint_img']
        joints_vis = sources['joint_vis']
        gt_joints = np.zeros((self.num_joints, 3, 2), dtype=np.float32)
        gt_joints[:, :, 0] = joints
        gt_joints[:, :, 1] = joints_vis
        bbox = list(sources['bbox'])
        xmin, ymin, xmax, ymax = bbox
        center, scale = _box_to_center_scale(
            xmin, ymin, xmax - xmin, ymax - ymin, 1, scale_mult=1)
        r = 0

        if self.data_aug:
            sf = 0.3
            scale = scale * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
            rf = 30
            r = np.clip(np.random.randn() * rf, -rf * 2, rf * 2) if random.random() <= 0.6 else 0

            if random.random() > 0.5:
                assert image.shape[2] == 3
                image = image[:, ::-1, :]
                imgwidth = image.shape[1]
                gt_joints = flip_joints_3d(gt_joints, imgwidth, ((1, 4), (2, 5), (3, 6), (14, 11), (15, 12), (16, 13)))
                center[0] = imgwidth - center[0] - 1

        inp_h, inp_w = self.size, self.size
        trans = get_affine_transform(center, scale, r, [inp_w, inp_h])
        image = cv2.warpAffine(image, trans, (int(inp_w), int(inp_h)), flags=cv2.INTER_LINEAR)

        # if self.data_aug:
        #     c_high = 1 + 0.2
        #     c_low = 1 - 0.2
        #     image[:, :, 0] = np.clip(image[:, :, 0] * random.uniform(c_low, c_high), 0, 255)
        #     image[:, :, 1] = np.clip(image[:, :, 1] * random.uniform(c_low, c_high), 0, 255)
        #     image[:, :, 2] = np.clip(image[:, :, 2] * random.uniform(c_low, c_high), 0, 255)

        image = self.transforms(image)
        
        joints = gt_joints
        # deal with joints visibility
        for i in range(self.num_joints):
            if joints[i, 0, 1] > 0.0:
                joints[i, 0:2, 0] = affine_transform(joints[i, 0:2, 0], trans)
        
        target_weight = np.ones((self.num_joints, 3), dtype=np.float32)
        target_weight[:, 0] = joints[:, 0, 1]
        target_weight[:, 1] = joints[:, 0, 1]
        target_weight[:, 2] = joints[:, 0, 1]

        target = np.zeros((self.num_joints, 3), dtype=np.float32)
        target[:, 0] = joints[:, 0, 0] / self.size
        target[:, 1] = joints[:, 1, 0] / self.size
        target[:, 2] = joints[:, 2, 0] / 2000 + 0.5

        target_weight[target[:, 0] < 0] = 0
        target_weight[target[:, 0] >= 1] = 0
        target_weight[target[:, 1] < 0] = 0
        target_weight[target[:, 1] >= 1] = 0
        target_weight[target[:, 2] < 0] = 0
        target_weight[target[:, 2] >= 1] = 0
        
        return image, target, target_weight, center, scale

def flip_joints_3d(joints_3d, width, joint_pairs):
    """Flip 3d joints.

    Parameters
    ----------
    joints_3d : numpy.ndarray
        Joints in shape (num_joints, 3, 2)
    width : int
        Image width.
    joint_pairs : list
        List of joint pairs.

    Returns
    -------
    numpy.ndarray
        Flipped 3d joints with shape (num_joints, 3, 2)

    """
    joints = joints_3d.copy()
    # flip horizontally
    joints[:, 0, 0] = width - joints[:, 0, 0] - 1
    # change left-right parts
    for pair in joint_pairs:
        joints[pair[0], :, 0], joints[pair[1], :, 0] = \
            joints[pair[1], :, 0], joints[pair[0], :, 0].copy()
        joints[pair[0], :, 1], joints[pair[1], :, 1] = \
            joints[pair[1], :, 1], joints[pair[0], :, 1].copy()

    joints[:, :, 0] *= joints[:, :, 1]
    return joints

def _box_to_center_scale(x, y, w, h, aspect_ratio=1.0, scale_mult=1.25):
    """Convert box coordinates to center and scale.
    adapted from https://github.com/Microsoft/human-pose-estimation.pytorch
    """
    pixel_std = 1
    center = np.zeros((2), dtype=np.float32)
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5

    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array(
        [w * 1.0 / pixel_std, h * 1.0 / pixel_std], dtype=np.float32)
    if center[0] != -1:
        scale = scale * scale_mult
    return center, scale

def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]

def get_3rd_point(a, b):
    """Return vector c that perpendicular to (a - b)."""
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    """Rotate the point by `rot_rad` degree."""
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result

def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords

def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0,
                         align=False):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale])

    scale_tmp = scale
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

def cam2pixel(cam_coord, f, c):
    x = cam_coord[:, 0] / (cam_coord[:, 2] + 1e-8) * f[0] + c[0]
    y = cam_coord[:, 1] / (cam_coord[:, 2] + 1e-8) * f[1] + c[1]
    z = cam_coord[:, 2]
    img_coord = np.concatenate((x[:, None], y[:, None], z[:, None]), 1)
    return img_coord

def pixel2cam(pixel_coord, f, c):
    x = (pixel_coord[:, 0] - c[0]) / f[0] * pixel_coord[:, 2]
    y = (pixel_coord[:, 1] - c[1]) / f[1] * pixel_coord[:, 2]
    z = pixel_coord[:, 2]
    cam_coord = np.concatenate((x[:, None], y[:, None], z[:, None]), 1)
    return cam_coord

def world2cam(world_coord, R, T):
    cam_coord = np.dot(R, world_coord - T)
    return cam_coord

def compute_similarity_transform(S1, S2):
    """
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    """
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert(S2.shape[1] == S1.shape[1]), (S1.shape, S2.shape)

    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = np.sum(X1**2)

    # 3. The outer product of X1 and X2.
    K = X1.dot(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    # Construct R.
    R = V.dot(Z.dot(U.T))

    # 5. Recover scale.
    scale = np.trace(R.dot(K)) / var1

    # 6. Recover translation.
    t = mu2 - scale * (R.dot(mu1))

    # 7. Error:
    S1_hat = scale * R.dot(S1) + t

    if transposed:
        S1_hat = S1_hat.T

    return S1_hat

def compute_similarity_transform_batch(S1, S2):
    """Batched version of compute_similarity_transform."""
    if S1.ndim == 2:
        S1_hat = compute_similarity_transform(S1.copy(), S2.copy())
    else:
        S1_hat = np.zeros_like(S1)
        for i in range(S1.shape[0]):
            S1_hat[i] = compute_similarity_transform(S1[i], S2[i])
    return S1_hat

def reconstruction_error(S1, S2):
    """Do Procrustes alignment and compute reconstruction error."""
    S1_hat = compute_similarity_transform_batch(S1, S2)
    return S1_hat

def bbox_xywh_to_xyxy(xywh):
    """Convert bounding boxes from format (x, y, w, h) to (xmin, ymin, xmax, ymax)

    Parameters
    ----------
    xywh : list, tuple or numpy.ndarray
        The bbox in format (x, y, w, h).
        If numpy.ndarray is provided, we expect multiple bounding boxes with
        shape `(N, 4)`.

    Returns
    -------
    tuple or numpy.ndarray
        The converted bboxes in format (xmin, ymin, xmax, ymax).
        If input is numpy.ndarray, return is numpy.ndarray correspondingly.

    """
    if isinstance(xywh, (tuple, list)):
        if not len(xywh) == 4:
            raise IndexError(
                "Bounding boxes must have 4 elements, given {}".format(len(xywh)))
        w, h = np.maximum(xywh[2] - 1, 0), np.maximum(xywh[3] - 1, 0)
        return (xywh[0], xywh[1], xywh[0] + w, xywh[1] + h)
    elif isinstance(xywh, np.ndarray):
        if not xywh.size % 4 == 0:
            raise IndexError(
                "Bounding boxes must have n * 4 elements, given {}".format(xywh.shape))
        xyxy = np.hstack((xywh[:, :2], xywh[:, :2] + np.maximum(0, xywh[:, 2:4] - 1)))
        return xyxy
    else:
        raise TypeError(
            'Expect input xywh a list, tuple or numpy.ndarray, given {}'.format(type(xywh)))


def bbox_xyxy_to_xywh(xyxy):
    """Convert bounding boxes from format (xmin, ymin, xmax, ymax) to (x, y, w, h).

    Parameters
    ----------
    xyxy : list, tuple or numpy.ndarray
        The bbox in format (xmin, ymin, xmax, ymax).
        If numpy.ndarray is provided, we expect multiple bounding boxes with
        shape `(N, 4)`.

    Returns
    -------
    tuple or numpy.ndarray
        The converted bboxes in format (x, y, w, h).
        If input is numpy.ndarray, return is numpy.ndarray correspondingly.

    """
    if isinstance(xyxy, (tuple, list)):
        if not len(xyxy) == 4:
            raise IndexError(
                "Bounding boxes must have 4 elements, given {}".format(len(xyxy)))
        x1, y1 = xyxy[0], xyxy[1]
        w, h = xyxy[2] - x1 + 1, xyxy[3] - y1 + 1
        return (x1, y1, w, h)
    elif isinstance(xyxy, np.ndarray):
        if not xyxy.size % 4 == 0:
            raise IndexError(
                "Bounding boxes must have n * 4 elements, given {}".format(xyxy.shape))
        return np.hstack((xyxy[:, :2], xyxy[:, 2:4] - xyxy[:, :2] + 1))
    else:
        raise TypeError(
            'Expect input xywh a list, tuple or numpy.ndarray, given {}'.format(type(xyxy)))


def bbox_clip_xyxy(xyxy, width, height):
    """Clip bounding box with format (xmin, ymin, xmax, ymax) to specified boundary.

    All bounding boxes will be clipped to the new region `(0, 0, width, height)`.

    Parameters
    ----------
    xyxy : list, tuple or numpy.ndarray
        The bbox in format (xmin, ymin, xmax, ymax).
        If numpy.ndarray is provided, we expect multiple bounding boxes with
        shape `(N, 4)`.
    width : int or float
        Boundary width.
    height : int or float
        Boundary height.

    Returns
    -------
    type
        Description of returned object.

    """
    if isinstance(xyxy, (tuple, list)):
        if not len(xyxy) == 4:
            raise IndexError(
                "Bounding boxes must have 4 elements, given {}".format(len(xyxy)))
        x1 = np.minimum(width - 1, np.maximum(0, xyxy[0]))
        y1 = np.minimum(height - 1, np.maximum(0, xyxy[1]))
        x2 = np.minimum(width - 1, np.maximum(0, xyxy[2]))
        y2 = np.minimum(height - 1, np.maximum(0, xyxy[3]))
        return (x1, y1, x2, y2)
    elif isinstance(xyxy, np.ndarray):
        if not xyxy.size % 4 == 0:
            raise IndexError(
                "Bounding boxes must have n * 4 elements, given {}".format(xyxy.shape))
        x1 = np.minimum(width - 1, np.maximum(0, xyxy[:, 0]))
        y1 = np.minimum(height - 1, np.maximum(0, xyxy[:, 1]))
        x2 = np.minimum(width - 1, np.maximum(0, xyxy[:, 2]))
        y2 = np.minimum(height - 1, np.maximum(0, xyxy[:, 3]))
        return np.hstack((x1, y1, x2, y2))
    else:
        raise TypeError(
            'Expect input xywh a list, tuple or numpy.ndarray, given {}'.format(type(xyxy)))