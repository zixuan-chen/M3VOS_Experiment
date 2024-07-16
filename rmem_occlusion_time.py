import numpy as np
import cv2
import os
from PIL import Image
from methods.RMem.aot_plus.tools.demo import color_palette, _palette, overlay, demo, main
from moviepy.editor import VideoFileClip, CompositeVideoClip
from VOSTest import VOSTest
import methods.RMem.aot_plus.dataloaders.video_transforms as tr
import importlib
from torchvision import transforms
from torch.utils.data import DataLoader
from methods.RMem.aot_plus.networks.models import build_vos_model
from methods.RMem.aot_plus.networks.engines import build_engine
from methods.RMem.aot_plus.utils.checkpoint import load_network
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt


engine_config = importlib.import_module('methods.RMem.aot_plus.configs.pre_vost')
cfg = engine_config.EngineConfig("default", "r50_aotl")

cfg.TEST_OUTPUT_PATH = "./output/occlusion/"
cfg.TEST_MIN_SIZE = None
cfg.TEST_MAX_SIZE = 480 * 1.3 * 800. / 480.
cfg.TEST_GPU_ID = 0
FRAME_RATE = 10
transform = transforms.Compose([
        tr.MultiRestrictSize(cfg.TEST_MIN_SIZE, cfg.TEST_MAX_SIZE,
                             cfg.TEST_FLIP, cfg.TEST_MULTISCALE,
                             cfg.MODEL_ALIGN_CORNERS),
        tr.MultiToTensor()
    ])
# 创建一个函数来生成带mask的帧
def make_masked_frame(frame, left=100, top=100, right=200, bottom=200):
    # 在masked_frame上绘制矩形mask，这里需要指定左上角和右下角的坐标
    # 假设mask的坐标为(left, top, right, bottom)
    # 创建一个与原始帧相同大小的黑色图像
    masked_frame = np.zeros_like(frame)
    # 将T时刻的帧复制到黑色图像上
    masked_frame += frame
    masked_frame = cv2.rectangle(np.ascontiguousarray(masked_frame), 
                                 (left, top), (right, bottom), (255, 255, 255), -1)
    return masked_frame


def make_masked_video_dataset(seq_name, 
                              start_time, 
                              duration,
                              left=100,
                              top=100,
                              right=200,
                              bottom=200):
    cfg.TEST_FRAME_PATH = '/methods/VOST/JPEGImages_10fps/%s/' % seq_name
    cfg.TEST_LABEL_PATH = "/methods/VOST/Annotations/%s/" % seq_name
    seq_images = np.sort(os.listdir(cfg.TEST_FRAME_PATH))
    image_root = "/".join(cfg.TEST_FRAME_PATH.split('/')[:-2])
    label_root = "/".join(cfg.TEST_LABEL_PATH.split('/')[:-2])
    seq_labels = [seq_images[0].replace('jpg', 'png')]
    seq_dataset = VOSTest(image_root,
                          label_root,
                          seq_name,
                          seq_images,
                          seq_labels,
                          transform=transform)
    step = int(start_time * FRAME_RATE)
    sample = seq_dataset[step]
    
    masked_frame = make_masked_frame(sample["current_img"], left, top, right, bottom)

    seq_dataset.insert_frames(step+1, int(duration*FRAME_RATE), masked_frame)
    return seq_dataset

def make_dataset_from_video(input_video_path, first_frame_path):
    pass

def aot_segment_video(seq_dataset: VOSTest):
    gpu_id = cfg.TEST_GPU_ID
    # Load pre-trained model
    print('Build AOT model.')
    model = build_vos_model(cfg.MODEL_VOS, cfg).cuda(gpu_id)

    print('Load checkpoint from {}'.format(cfg.TEST_CKPT_PATH))
    model, _ = load_network(model, cfg.TEST_CKPT_PATH, gpu_id)

    print('Build AOT engine.')
    engine = build_engine(cfg.MODEL_ENGINE,
                          phase='eval',
                          aot_model=model,
                          gpu_id=gpu_id,
                          long_term_mem_gap=cfg.TEST_LONG_TERM_MEM_GAP)
    output_root = cfg.TEST_OUTPUT_PATH
    output_mask_root = os.path.join(output_root, 'pred_masks')
    if not os.path.exists(output_mask_root):
        os.makedirs(output_mask_root)
    
    seq_name = seq_dataset.seq_name
    image_seq_root = os.path.join(seq_dataset.image_root, seq_name)
    output_mask_seq_root = os.path.join(output_mask_root, seq_name)
    if not os.path.exists(output_mask_seq_root):
        os.makedirs(output_mask_seq_root)
    print('Build a dataloader for sequence {}.'.format(seq_name))
    seq_dataloader = DataLoader(seq_dataset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=cfg.TEST_WORKERS,
                                pin_memory=True)
    num_frames = len(seq_dataset)
    max_gap = int(round(num_frames / 30))
    gap = max(max_gap, 5)
    print(gap)
    engine.long_term_mem_gap = gap

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_video_path = os.path.join(
        output_root, '{}_{}fps.avi'.format(seq_name, FRAME_RATE))

    print('Start the inference of sequence {}:'.format(seq_name))
    model.eval()
    engine.restart_engine()
    with torch.no_grad():
        time_start = 0
        for frame_idx, samples in enumerate(seq_dataloader):
            # if time_start != 0:
            #     print(time() - time_start)        
            # time_start = time()
            sample = samples[0]
            img_name = sample['meta']['current_name'][0]

            obj_nums = sample['meta']['obj_num']
            output_height = sample['meta']['height']
            output_width = sample['meta']['width']
            obj_idx = sample['meta']['obj_idx']

            obj_nums = [int(obj_num) for obj_num in obj_nums]
            obj_idx = [int(_obj_idx) for _obj_idx in obj_idx]

            current_img = sample['current_img']
            current_img = current_img.cuda(gpu_id, non_blocking=True)

            if frame_idx == 0:
                videoWriter = cv2.VideoWriter(
                    output_video_path, fourcc, FRAME_RATE,
                    (int(output_width), int(output_height)))
                print(
                    'Object number: {}. Inference size: {}x{}. Output size: {}x{}.'
                    .format(obj_nums[0],
                            current_img.size()[2],
                            current_img.size()[3], int(output_height),
                            int(output_width)))
                current_label = sample['current_label'].cuda(
                    gpu_id, non_blocking=True).float()
                current_label = F.interpolate(current_label,
                                                size=current_img.size()[2:],
                                                mode="nearest")
                # add reference frame
                engine.add_reference_frame(current_img,
                                            current_label,
                                            frame_step=0,
                                            obj_nums=obj_nums)
            else:
                print('Processing image {}...'.format(img_name))
                # predict segmentation
                engine.match_propogate_one_frame(current_img)
                pred_logit = engine.decode_current_logits(
                    (output_height, output_width))
                pred_prob = torch.softmax(pred_logit, dim=1)
                pred_label = torch.argmax(pred_prob, dim=1,
                                            keepdim=True).float()
                _pred_label = F.interpolate(pred_label,
                                            size=engine.input_size_2d,
                                            mode="nearest")
                # update memory
                engine.update_memory(_pred_label)

                # save results
                input_image_path = os.path.join(image_seq_root, img_name)
                output_mask_path = os.path.join(
                    output_mask_seq_root,
                    img_name.split('.')[0] + '.png')

                pred_label = Image.fromarray(
                    pred_label.squeeze(0).squeeze(0).cpu().numpy().astype(
                        'uint8')).convert('P')
                pred_label.putpalette(_palette)
                pred_label.save(output_mask_path)

                input_image = Image.open(input_image_path)

                overlayed_image = overlay(
                    np.array(input_image, dtype=np.uint8),
                    np.array(pred_label, dtype=np.uint8), color_palette)
                videoWriter.write(overlayed_image[..., [2, 1, 0]])        

    print('Save a visualization video to {}.'.format(output_video_path))
    videoWriter.release()

if __name__ == "__main__":
    seq_name = "7866_squeeze_bag"
    start_time = 5
    left=600
    top=50
    right=1000
    bottom=400
    duration = start_time*FRAME_RATE
    seq_dataset = make_masked_video_dataset(seq_name, start_time, duration,
                              left, top, right, bottom)
    aot_segment_video(seq_dataset)
