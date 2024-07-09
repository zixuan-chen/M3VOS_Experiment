import cv2
import os
import numpy as np
from PIL import Image

def overlay_masks_on_video(video_path, masks_folder, output_path):
    # 打开原视频
    video_capture = cv2.VideoCapture(video_path)
 

    if not video_capture.isOpened():
        print("Error: Could not open video.")
        return

    # 获取视频的帧宽、帧高和帧率
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_capture.get(cv2.CAP_PROP_FPS)

    # 设置输出视频的编解码器和参数
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    print("Save into:", output_path)

    # 获取掩码文件列表，并按顺序排序
    mask_files = sorted([f for f in os.listdir(masks_folder) if f.endswith('.png')])
    
    frame_index = 0
    mask_index = 0

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # 如果当前帧有对应的掩码，则加载掩码
        if mask_index < len(mask_files):
            mask_file = mask_files[mask_index]
            mask_frame_number = int(mask_file.split('.')[0])
            if frame_index == mask_frame_number:
                mask_path = os.path.join(masks_folder, mask_file)
                mask = np.array(Image.open(mask_path))  # 读取为灰度图

                if mask is not None:
                    # 将掩码转换为0-255的范围
                    mask = (mask * 255).astype(np.uint8)

                    # 创建一个三通道的彩色掩码
                    colored_mask = np.zeros_like(frame)
                    colored_mask[:, :, 1] = mask  # 将掩码的1值转换为绿色

                    # 将掩码叠加到原视频帧上
                    frame = cv2.addWeighted(frame, 1, colored_mask, 0.5, 0)

                mask_index += 1

        video_writer.write(frame)
        frame_index += 1

    video_capture.release()
    video_writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_id = "4891"
    video_path = f'/data/zixuan/sthsthv2/Bending_something_so_that_it_deforms/{video_id}.webm'
    masks_folder = f'/home/chenzixuan/myVOS/output_dir/{video_id}'
    output_path = f'/home/chenzixuan/myVOS/{video_id}.mp4'

    overlay_masks_on_video(video_path, masks_folder, output_path)