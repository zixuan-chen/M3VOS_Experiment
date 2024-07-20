from PIL import Image
from moviepy.editor import VideoFileClip
import os 

def rotate_and_save_image_if_tall(image_path):
    """
    如果图片的宽度小于高度，则逆时针旋转90度并替换原图。
    
    参数:
    image_path: 图片的路径。
    """
    # 打开图片
    img = Image.open(image_path)
    
    # 检查图片的宽度是否大于高度
    if img.height > img.width:
        # 如果是，逆时针旋转90度
        img = img.rotate(90, expand=True)
        # 保存替换原图
        img.save(image_path)


def rotate_video_if_tall(video_path):
    """
    如果视频的宽度小于高度，则逆时针旋转90度。
    
    参数:
    video_path: 视频的路径。
    """
    # 使用moviepy打开视频
    clip = VideoFileClip(video_path)
    
    # 检查视频的宽度是否小于高度
    if clip.size[0] < clip.size[1]:
        # 如果是，逆时针旋转90度
        rotated_clip = clip.rotate(90)
        video_path_tmp = video_path.split('.')[0] + '_tmp' + '.mp4'
        print("ROTA the video:",  video_path_tmp)
        # 保存修改后的视频到原路径，也可以选择新路径
        rotated_clip.write_videofile(video_path_tmp, codec="libx264", audio_codec="aac")

# def main_worker():




if __name__ == "__main__":
    # rotate_and_save_image_if_tall("/home/chenzixuan/jiaxin/DeformVOS/tmp/mask_v.png")
    rotate_video_if_tall("/home/chenzixuan/jiaxin/DeformVOS/tmp/seq_5_cup.mp4")

 
 