import vot
import sys
# import glob
import copy
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

from PIL import Image
from pathlib import Path
from omegaconf import OmegaConf
from collections import OrderedDict
from argparse import Namespace

# fix seed
np.random.seed(42)
torch.manual_seed(42)

# specific imports
sys.path.append('/home/v-chrisfan/work/vector_backup/work/epi_mem_vos_101501')
sys.path.append('/home/v-chrisfan/work/vector_backup/work/epi_mem_vos_101501/sources')
from datasets.video_transforms import make_video_transforms
from models import build_model_vost
from util.misc import NestedTensor
import torch.nn.functional as F

# Testing setting
# for visualization
is_save_mask = True
save_mask_type = 'merged' # or 'splitted' splitted will save each object mask separately
outputs_vis = '/home/v-chrisfan/work/vector_backup/work/epi_mem_vos_101501/vots_challenge/votst_test_tracker/outputs_final'
downgrade_fps_ratio = 2 # 1: keep original fps, 2: downgrade to half fps, 3: downgrade to 1/3 fps

#########################################
# for visualization
def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

def save_mask(mask, img_path):
    if np.max(mask) > 255:
        raise ValueError('Maximum id pixel value is 255')
    mask_img = Image.fromarray(mask.astype(np.uint8))
    mask_img.putpalette(color_map().flatten().tolist())
    mask_img.save(img_path)

def visualizer(raw_masks, img_path):
    raw_masks_vis = copy.deepcopy(raw_masks)

    if save_mask_type == 'merged':
        for i in range(len(raw_masks_vis)):
            raw_masks_vis[i] = raw_masks_vis[i] * (i + 1)  # TODO: check this

        raw_masks_stacked = np.stack(raw_masks_vis, axis=0)
        raw_masks_stacked = np.sum(raw_masks_stacked, axis=0) # max merging
        save_mask(raw_masks_stacked, img_path)

    elif save_mask_type == 'splitted':
        for idx, mask in enumerate(raw_masks_vis):
            save_mask(mask, img_path.replace('.png', f'_{idx}.png'))

#########################################

def fetch_model():
    args_from_cli = Namespace(
        config_path='config/vost_multi_scale_memory.yaml',
        opts=[
            'resume=checkpoints/tamvt_vost.pth',
            "tasks.names=['vost']",
            'epochs=150',
            'eval_skip=5',
            'lr_drop=150',
            'num_workers=0',
            'train_flags.print_freq=2',
            'resolution=624',
            'model.vost.video_max_len=12',
            'output_dir=outputs/trial/',
            'sted=False',
            'model.use_score_per_frame=False',
            'aux_loss=True',
            'eval_flags.vost.eval_first_window_only=False',
            'eval=True',
            'data.vost.use_test_transform=False',
            'eval_flags.plot_pred=True',
            'model.vost.memory.clip_length=2',
            'model.vost.memory.bank_size=9',
            'model.name.tubedetr=tubedetr_vost_multi_scale_memory_ms',
            'model.name.transformer=transformer_vost_multi_scale_memory_last_layer_too_ms',
            'use_rand_reverse=True',
            'use_ignore_threshold=True',
            'model.vost.memory.rel_time_encoding=True',
            'model.vost.memory.rel_time_encoding_type=embedding_dyna_mul',
            'eval_flags.vost.vis_only_no_cache=True',
            'eval_flags.vost.vis_only_pred_mask=True',
            'backbone=resnet50'])

    cfg_from_default = OmegaConf.load(args_from_cli.config_path)
    cfg_from_tubedetr_base = OmegaConf.load(cfg_from_default._BASE_)
    cfg_from_cli = OmegaConf.from_cli(args_from_cli.opts)

    args = OmegaConf.merge(cfg_from_tubedetr_base, cfg_from_default, cfg_from_cli)

    model, criterion, weight_dict = build_model_vost(args)
    model.eval()

    assert args.resume

    print("resuming from", args.resume)
    checkpoint = torch.load(args.resume, map_location="cpu")

    model.load_state_dict(checkpoint["model"])

    return model, args


class TAMVT:
    def __init__(self, args, device, n_objects, memory_images, memory_masks, model):
        self.args = args
        self.model = model.to(device)
        self.device = device

        # variables needed
        self.n_objects = n_objects
        self.window_step_size = args.model.vost.memory.clip_length

        self.init_memory(memory_images, memory_masks)

    @torch.no_grad()
    def init_memory(self, memory_images, memory_masks):
        self.memory_encoded = [OrderedDict() for _ in range(self.n_objects)]

        self.memory = []
        for i_object in range(self.n_objects):
            _memory = OrderedDict({
                0: {
                    "image": NestedTensor(memory_images.tensors, memory_images.mask).to(self.device),
                    "mask": NestedTensor(
                        memory_masks.tensors[:, i_object: i_object + 1],
                        memory_masks.mask[:, i_object: i_object + 1]
                    ).to(self.device),
                }
            })
            self.memory.append(_memory)

    @torch.no_grad()
    def step(self, i_object, samples_window):

        samples_window = samples_window.to(self.device)

        memory_cache_window, memory_encoded, memory = self.model(
            "vost",
            samples_window,
            [self.window_step_size],
            encode_and_save=True,
            samples_fast=None,
            memory_encoded=self.memory_encoded[i_object],
            memory=self.memory[i_object],
        )
        outputs_window = self.model(
            "vost",
            samples_window,
            [self.window_step_size],
            encode_and_save=False,
            memory_cache=memory_cache_window,
        )

        # propagate memory
        _pred_last = F.interpolate(
            outputs_window['pred_masks'][-1][None], size=samples_window.tensors.shape[-2:],
            mode="bilinear", align_corners=False
        ).sigmoid()

        # take the last image
        _mem_image_forward = samples_window.tensors[-1].unsqueeze(1)  # (C, N, H, W)
        _mem_image_forward = NestedTensor.from_tensor_list([_mem_image_forward.to(self.device)])

        _mem_mask_forward = _pred_last.detach()  # (C, N, H, W)
        _mem_mask_forward = NestedTensor.from_tensor_list([_mem_mask_forward.to(self.device)])

        if len(self.memory[i_object]) >= self.args.model.vost.memory.bank_size and self.args.model.vost.memory.bank_size != 1:
            _key_to_remove = [*self.memory[i_object].keys()][1]  # keep first frame fixed
            assert _key_to_remove != 0

            # remove entries from memory and encoded memory
            del self.memory[i_object][_key_to_remove]
            del self.memory_encoded[i_object][_key_to_remove]

            self.memory[i_object].update({max(self.memory[i_object].keys()) + 1: {
                "image": _mem_image_forward, "mask": _mem_mask_forward
            }})
        elif len(self.memory[i_object]) >= self.args.model.vost.memory.bank_size and self.args.model.vost.memory.bank_size == 1:
            pass
        else:
            self.memory[i_object].update({max(self.memory[i_object].keys()) + 1: {
                "image": _mem_image_forward, "mask": _mem_mask_forward
            }})

        return outputs_window

    @torch.no_grad()
    def get_predictions(self, samples_window_nested_tensor, original_size):
        # step
        outputs_window = [
            self.step(i_object, samples_window_nested_tensor)
            for i_object in range(1)
        ]

        # upsample output
        pred_mask_upsamp = [
            F.interpolate(
                _outputs_window['pred_masks'], size=original_size, mode="bilinear", align_corners=False)
            for _outputs_window in outputs_window
        ]
        pred_mask_upsamp = [
            (_pred_mask_upsamp.sigmoid() > 0.5)
            for _pred_mask_upsamp in pred_mask_upsamp
        ]

        return pred_mask_upsamp

def decouple_mask(pred_mask_upsamp):
    """
    If the object appears in the first object mask, then it should not appear in the second or above object mask.
    This function is to fix the issue. 
    """
    pred_mask_upsamp_new = copy.deepcopy(pred_mask_upsamp)
    pred_mask_upsamp_new = pred_mask_upsamp_new[::-1]
    for i_image in range(len(pred_mask_upsamp_new[0])):
        for i_object in range(1, len(pred_mask_upsamp_new)):
            for j in range(i_object):
                pred_mask_upsamp_new[i_object][i_image] = ( pred_mask_upsamp_new[i_object][i_image].int()  * (1 - pred_mask_upsamp_new[j][i_image].int()) ).bool()
    pred_mask_upsamp_new = pred_mask_upsamp_new[::-1]

    # if the mask is empty, there will be some error in the following steps
    # create dummy mask if empty
    for i_image in range(len(pred_mask_upsamp_new[0])):
        for i_object in range(len(pred_mask_upsamp_new)):
            if pred_mask_upsamp_new[i_object][i_image].sum() == 0:
                pred_mask_upsamp_new[i_object][i_image][0][0][:2] = True

    return pred_mask_upsamp_new

# VOT init
handle = vot.VOT("mask", multiobject=True)
objects = handle.objects()

# fetch inital frame
imagefile = handle.frame()
print("First frame path: ", imagefile)

image_initial = np.array(Image.open(imagefile).convert('RGB'))
h = image_initial.shape[0]
w = image_initial.shape[1]

# fetch object masks
num_objects = len(objects)
masks_objects = []
for _i in range(num_objects):
    _mask = np.zeros((h, w), dtype=np.uint8)
    _mask[:objects[_i].shape[0], :objects[_i].shape[1]] = objects[_i]
    masks_objects.append(_mask)

#### FETCH MODEL AND ARGS
all_model_list = []
for _i in range(num_objects):
    model_list = []
    for i in range(downgrade_fps_ratio):
        model, args = fetch_model()
        model_list.append(model)
    all_model_list.append(model_list)

#### PRE-PROCESSING

# init transform
transforms = make_video_transforms("val", cautious=True, resolution=624)

all_tracker_list = []
for obj_i in range(num_objects):
    masks_objects_single = [masks_objects[obj_i]]
    # pre-process initial frame and mask
    masks = np.stack(masks_objects_single)
    masks = torch.as_tensor(masks, dtype=torch.uint8)
    assert masks.shape[0] == 1

    frames = image_initial[None]
    target = {'masks': masks}

    # transform initial frame along with masks
    images, targets = transforms(frames, copy.deepcopy([target]))

    # obtain memory images and masks for tracker
    memory_images, memory_masks = [], []
    memory_images.append(images[:, 0])
    memory_masks.append(targets[0]['masks'].float())

    memory_images = torch.stack(memory_images, dim=1)  # (C, N, H, W)
    memory_masks = torch.stack(memory_masks, dim=1)  # (C, N, H, W)

    memory_images_nested_tensor = NestedTensor.from_tensor_list([memory_images])
    memory_masks_nested_tensor = NestedTensor.from_tensor_list([memory_masks])

    #### INIT TRACKER
    tracker_list = []
    for i in range(downgrade_fps_ratio):
        tracker_list.append(TAMVT(args, 'cuda', 1, memory_images_nested_tensor, memory_masks_nested_tensor, all_model_list[obj_i][i]))

    all_tracker_list.append(tracker_list)

# trackers = [TAMVTTracker(image, object) for object in objects]

# version 2
os.makedirs(outputs_vis, exist_ok=True)

count = 0
prediction_buffer_list = [[] for _ in range(downgrade_fps_ratio)]
clip_length = args.model.vost.memory.clip_length
clip_length_scaled = clip_length * downgrade_fps_ratio

while True:
    with torch.no_grad():

        print("Count: ", count)
        p_image = handle.frame()
        print("Current frame path:", p_image)
        frame_dir_path = os.path.dirname(p_image)
        p_frame_name = os.path.basename(p_image)
        video_name = os.path.basename(os.path.dirname(frame_dir_path))

        # check if this is the last frame
        if not p_image:
            break

        # check buffer
        prediction_buffer = copy.deepcopy(prediction_buffer_list[count % downgrade_fps_ratio])
        print('check lenght of buffer: ', len(prediction_buffer))
        if len(prediction_buffer) > 0:
            print("Use buffer!")
            handle.report(prediction_buffer[0])
            prediction_buffer_list[count % downgrade_fps_ratio] = prediction_buffer[1:]
            print("Buffer: ", prediction_buffer_list)
            
        # if nothing in buffer, perform prediction
        else:
            print("Do prediction!!")
            # collecting next frames into clip
            all_frame_list = sorted(os.listdir(frame_dir_path))
            current_idx = all_frame_list.index(p_frame_name)

            if current_idx + clip_length_scaled > len(all_frame_list): 
                frame_names_in_clip = all_frame_list[current_idx::downgrade_fps_ratio]
            else:
                frame_names_in_clip = all_frame_list[current_idx:current_idx + clip_length_scaled:downgrade_fps_ratio]
            
            print("Images in clip:", frame_names_in_clip)

            samples_window = []
            for frame_name in frame_names_in_clip:
                new_image = np.array(Image.open(os.path.join(frame_dir_path, frame_name)).convert('RGB'))
                samples_window.append(new_image)

            # image list --> transform --> NestedTensor
            samples_window_transformed, _ = transforms(np.stack(samples_window), None)
            samples_window_nested_tensor = NestedTensor.from_tensor_list([samples_window_transformed])

            # obtain predictions
            # pred_mask_upsamp => (N obj, clip size, 1, (mask size))
            pred_mask_upsamp = []
            for i_object in range(num_objects):
                tracker_list = all_tracker_list[i_object]
                prediction = tracker_list[count % downgrade_fps_ratio].get_predictions(samples_window_nested_tensor, original_size=(h, w))
                assert len(prediction) == 1
                pred_mask_upsamp.append(prediction[0])
            
            # pred_mask_upsamp = tracker_list[count % downgrade_fps_ratio].get_predictions(samples_window_nested_tensor, original_size=(h, w))

            # fix the second and third object will for some reason include the first object
            # decouple only happens in output side, the frame feed into the model is still the same
            pred_mask_upsamp_decoupled = decouple_mask(pred_mask_upsamp)
            _pred_mask_upsamp_decoupled = copy.deepcopy(pred_mask_upsamp_decoupled)
            # report first image in the clip and save the rest into prediction buffer
            handle.report([
                _pred_mask_upsamp_decoupled[i_object].cpu().numpy()[0][0].astype(np.uint8) # report type should be unit8
                for i_object in range(num_objects)
            ])

            # save mask
            if is_save_mask:
                _pred_mask_upsamp_decoupled = copy.deepcopy(pred_mask_upsamp_decoupled)
                os.makedirs(os.path.join(outputs_vis, video_name), exist_ok=True)
                for i_image in range(len(frame_names_in_clip)):
                    visualizer([_pred_mask_upsamp_decoupled[i_object].cpu().numpy()[i_image][0].astype(np.uint8) for i_object in range(num_objects)], 
                            os.path.join(outputs_vis, video_name, frame_names_in_clip[i_image].replace('jpg', 'png')))

            _pred_mask_upsamp_decoupled = copy.deepcopy(pred_mask_upsamp_decoupled)
            for i_image in range(1, clip_length):
                prediction_buffer.append([
                    _pred_mask_upsamp_decoupled[i_object].cpu().numpy()[i_image][0].astype(np.uint8)
                    for i_object in range(num_objects)
                ])
            print("Buffer length: ", len(prediction_buffer))

        count += 1

