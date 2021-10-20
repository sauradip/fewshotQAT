import os
import torch
import torch.nn.functional as F
import yaml
import copy
from ast import literal_eval
from typing import Callable, Iterable, List, TypeVar
import torch.distributed as dist
from typing import Tuple
import argparse
from scipy import ndimage
import itertools
import numpy as np
import pandas as pd
from gtad_lib import opts

A = TypeVar("A")
B = TypeVar("B")

test_class = ['Hurling', 'Polishing forniture', 'BMX', 'Riding bumper cars', 'Starting a campfire', 'Walking the dog', 'Preparing salad', 'Plataform diving', 'Breakdancing', 'Camel ride', 'Hand car wash', 'Making an omelette', 'Shuffleboard', 'Calf roping', 'Shaving legs', 'Snatch', 'Cleaning sink', 'Rope skipping', 'Drinking coffee', 'Pole vault']
actions = pd.read_csv("/media/phd/SAURADIP5TB/ACLPT/activitynet_annotations/action_name.csv")
cnames = list(set(test_class).intersection(set(actions.action.values.tolist())))
        
opt = opts.parse_opt()
opt = vars(opt)

def main_process(args: argparse.Namespace) -> bool:
    if args.distributed:
        rank = dist.get_rank()
        if rank == 0:
            return True
        else:
            return False
    else:
        return True


def setup(args: argparse.Namespace,
          rank: int,
          world_size: int) -> None:
    """
    Used for distributed learning
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(args.port)

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup() -> None:
    """
    Used for distributed learning
    """
    dist.destroy_process_group()


def find_free_port() -> int:
    """
    Used for distributed learning
    """
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def map_(fn: Callable[[A], B], iter: Iterable[A]) -> List[B]:
    """
    Used for multiprocessing
    """
    return list(map(fn, iter))


def get_model_dir(args: argparse.Namespace) -> str:
    """
    Obtain the directory to save/load the model
    """
    path = os.path.join('/media/phd/SAURADIP5TB/dataset/RePRi/ckpt/model_ckpt',
                        args.train_name,
                        f'split={args.train_split}',
                        'model',
                        f'pspnet_{args.arch}{args.layers}',
                        f'smoothing={args.smoothing}',
                        f'mixup={args.mixup}')
    return path


def to_one_hot(mask: torch.tensor,
               num_classes: int) -> torch.tensor:
    """
    inputs:
        mask : shape [n_task, shot, h, w]
        num_classes : Number of classes

    returns :
        one_hot_mask : shape [n_task, shot, num_class, h, w]
    """
    n_tasks, shot, h, w = mask.size()
    one_hot_mask = torch.zeros(n_tasks, shot, num_classes, h, w).cuda()
    new_mask = mask.unsqueeze(2).clone()
    # print()
    new_mask[torch.where(new_mask == 0)] = 0  # Ignore_pixels are anyways filtered out in the losses
    one_hot_mask.scatter_(2, new_mask, 1).long()
    return one_hot_mask


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def batch_intersectionAndUnionGPU(logits: torch.Tensor,
                                  target: torch.Tensor,
                                  num_classes: int,
                                  subcls: List[int],
                                  vid_name : str,
                                  ignore_index=255
                                   ) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
    """
    inputs:
        logits : shape [n_task, shot, num_class, h, w]
        target : shape [n_task, shot, H, W]
        num_classes : Number of classes

    returns :
        area_intersection : shape [n_task, shot, num_class]
        area_union : shape [n_task, shot, num_class]
        area_target : shape [n_task, shot, num_class]
    """
    n_task, shots, num_classes, h, w = logits.size() ## h, w are low dimension
    H, W = target.size()[-2:] ## H,W are high dimension

    logits = F.interpolate(logits.view(n_task * shots, num_classes, h, w),
                           size=(H, W), mode='bilinear', align_corners=True).view(n_task, shots, num_classes, H, W) #### interpolates from h -> H and w -> W
    preds = logits.argmax(2)  # [n_task, shot, H, W]
    # print("from util", logpredsits)
    # print(logits)


    # filtered_seq_int = preds.detach().cpu().numpy().tolist()
    # start_pt = filtered_seq_int.index(1)
    # end_pt = len(filtered_seq_int) - 1 - filtered_seq_int[::-1].index(1)
    # print("vid_name", vid_name)
    findTAL(logits,target,subcls, vid_name)

    n_tasks, shot, num_classes, H, W = logits.size()
    area_intersection = torch.zeros(n_tasks, shot, num_classes)
    area_union = torch.zeros(n_tasks, shot, num_classes)
    area_target = torch.zeros(n_tasks, shot, num_classes)
    # for task in range(n_tasks):
    #     for shot in range(shots):
    #         # print(preds[task][shot].size())
    #         i, u, t = intersectionAndUnionGPU(preds[task][shot], target[task][shot],
    #                                           num_classes, ignore_index=ignore_index) # i,u, t are of size()
    #         area_intersection[task, shot, :] = i
    #         area_union[task, shot, :] = u
    #         area_target[task, shot, :] = t
    return area_intersection, area_union, area_target

def findTAL(logits,target,sub_cls, vid_name):
    n_tasks, shot, num_classes, H, W = logits.size()
    temp_list=[]
    for task in range(n_tasks):
        for shot1 in range(shot):
            # print("clss",cnames[sub_cls[task][shot1]])
            # print("query_vid", vid_name[0])
            fg = logits[task][shot1][0].detach().cpu().numpy()
            bg = logits[task][shot1][1].detach().cpu().numpy()
            
            # thres = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
            thres = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
            # thres = [x / 100.0 for x in range(0, 100, 5)]
            # print(thres)
            for i in thres:
                fg_thres = fg > i
                integer_map = map(int,fg_thres)
                filtered_seq_int = list(integer_map)
                filled_fg = ndimage.binary_fill_holes(filtered_seq_int).astype(int).tolist()
                if 1 in filled_fg:            
                    start_pt = filled_fg.index(1)
                    end_pt = len(filled_fg) - 1 - filled_fg[::-1].index(1)
                    if (end_pt/100) > (start_pt/100) and ((end_pt - start_pt)/100) > 0.1 :
                        reg_vals = np.mean(fg[start_pt:end_pt])
                        conf_vals = np.amax(fg[start_pt:end_pt])
                        # print("conf",conf_vals)
                        temp_list.append([start_pt/100,end_pt/100,conf_vals,conf_vals])
                # else:
                    
            # print("temp_list", temp_list)
            # print(target[task][shot1].detach().cpu().numpy().tolist())
    new_props = np.stack(temp_list)
    video_name = vid_name[0]
    col_name = ["xmin", "xmax", "clr_score", "reg_socre"]
    new_df = pd.DataFrame(new_props, columns=col_name)
    

    

    new_df.to_csv(opt["output"]+"/results1/" + video_name + ".csv", index=False)
            # tgt = target[task][shot1].detach().cpu().numpy().tolist()
            # tgt = list(itertools.chain.from_iterable(tgt))
            # start_pt = tgt.index(1)
            # end_pt = len(tgt) - 1 - tgt[::-1].index(1)
            # print("target_list", start_pt,end_pt)


            # print(logits[task][shot1][0].size())
def intersectionAndUnionGPU(preds: torch.tensor,
                            target: torch.tensor,
                            num_classes: int,
                            ignore_index=255) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
    """
    inputs:
        preds : shape [H, W]
        target : shape [H, W]
        num_classes : Number of classes

    returns :
        area_intersection : shape [num_class]
        area_union : shape [num_class]
        area_target : shape [num_class]
    """
    ignore_index = 0
    assert (preds.dim() in [1, 2, 3])
    assert preds.shape == target.shape
    preds = preds.view(-1)
    
    # print("pred start",start_pt,end_pt)
    # print("pred end",end_pt)
    target = target.view(-1)
    # filtered_seq_int = target.detach().cpu().numpy().tolist()
    # start_pt = filtered_seq_int.index(1)
    # end_pt = len(filtered_seq_int) - 1 - filtered_seq_int[::-1].index(1)
    # print("target start",start_pt,end_pt)
    # print("target e nd",end_pt)
    
    preds[target == ignore_index] = ignore_index
    intersection = preds[preds == target]

    # Addind .float() becausue histc not working with long() on CPU
    area_intersection = torch.histc(intersection.float(), bins=num_classes, min=0, max=num_classes-1)
    area_output = torch.histc(preds.float(), bins=num_classes, min=0, max=num_classes-1)
    area_target = torch.histc(target.float(), bins=num_classes, min=0, max=num_classes-1)
    area_union = area_output + area_target - area_intersection
    # print(torch.unique(intersection))
    return area_intersection, area_union, area_target


# ======================================================================================================================
# ======== All following helper functions have been borrowed from from https://github.com/Jia-Research-Lab/PFENet ======
# ======================================================================================================================

class CfgNode(dict):
    """
    CfgNode represents an internal node in the configuration tree. It's a simple
    dict-like container that allows for attribute-based access to keys.
    """

    def __init__(self, init_dict=None, key_list=None, new_allowed=False):
        # Recursively convert nested dictionaries in init_dict into CfgNodes
        init_dict = {} if init_dict is None else init_dict
        key_list = [] if key_list is None else key_list
        for k, v in init_dict.items():
            if type(v) is dict:
                # Convert dict to CfgNode
                init_dict[k] = CfgNode(v, key_list=key_list + [k])
        super(CfgNode, self).__init__(init_dict)

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value

    def __str__(self):
        def _indent(s_, num_spaces):
            s = s_.split("\n")
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(num_spaces * " ") + line for line in s]
            s = "\n".join(s)
            s = first + "\n" + s
            return s

        r = ""
        s = []
        for k, v in sorted(self.items()):
            seperator = "\n" if isinstance(v, CfgNode) else " "
            attr_str = "{}:{}{}".format(str(k), seperator, str(v))
            attr_str = _indent(attr_str, 2)
            s.append(attr_str)
        r += "\n".join(s)
        return r

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, super(CfgNode, self).__repr__())


def _decode_cfg_value(v):
    if not isinstance(v, str):
        return v
    try:
        v = literal_eval(v)
    except ValueError:
        pass
    except SyntaxError:
        pass
    return v


def _check_and_coerce_cfg_value_type(replacement, original, key, full_key):
    original_type = type(original)
    replacement_type = type(replacement)

    # The types must match (with some exceptions)
    if replacement_type == original_type:
        return replacement

    def conditional_cast(from_type, to_type):
        if replacement_type == from_type and original_type == to_type:
            return True, to_type(replacement)
        else:
            return False, None

    casts = [(tuple, list), (list, tuple)]
    try:
        casts.append((str, unicode))  # noqa: F821
    except Exception:
        pass

    for (from_type, to_type) in casts:
        converted, converted_value = conditional_cast(from_type, to_type)
        if converted:
            return converted_value

    raise ValueError(
        "Type mismatch ({} vs. {}) with values ({} vs. {}) for config "
        "key: {}".format(
            original_type, replacement_type, original, replacement, full_key
        )
    )


def load_cfg_from_cfg_file(file: str):
    cfg = {}
    assert os.path.isfile(file) and file.endswith('.yaml'), \
        '{} is not a yaml file'.format(file)

    with open(file, 'r') as f:
        cfg_from_file = yaml.safe_load(f)

    for key in cfg_from_file:
        for k, v in cfg_from_file[key].items():
            cfg[k] = v

    cfg = CfgNode(cfg)
    return cfg


def merge_cfg_from_list(cfg: CfgNode,
                        cfg_list: List[str]):
    new_cfg = copy.deepcopy(cfg)
    assert len(cfg_list) % 2 == 0, cfg_list
    for full_key, v in zip(cfg_list[0::2], cfg_list[1::2]):
        subkey = full_key.split('.')[-1]
        assert subkey in cfg, 'Non-existent key: {}'.format(full_key)
        value = _decode_cfg_value(v)
        value = _check_and_coerce_cfg_value_type(
            value, cfg[subkey], subkey, full_key
        )
        setattr(new_cfg, subkey, value)

    return new_cfg
