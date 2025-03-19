from PIL import Image
from moviepy.editor import VideoFileClip
import os 

def rotate_and_save_image_if_tall(image_path):
    img = Image.open(image_path)
    if img.height > img.width:
        img = img.rotate(90, expand=True)
        img.save(image_path)


def rotate_video_if_tall(video_path):
    clip = VideoFileClip(video_path)
    
    if clip.size[0] < clip.size[1]:
 
        rotated_clip = clip.rotate(90)
        video_path_tmp = video_path.split('.')[0] + '_tmp' + '.mp4'
        print("ROTA the video:",  video_path_tmp)

        rotated_clip.write_videofile(video_path_tmp, codec="libx264", audio_codec="aac")





 
 