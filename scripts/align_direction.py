from PIL import Image
from moviepy.editor import VideoFileClip
import os 

def rotate_and_save_image_if_tall(image_path):
    """
    If the image's width is less than its height, rotate it counterclockwise by 90 degrees and overwrite the original image.
    
    Parameters:
    image_path: Path to the image.
    """
    # Open the image
    img = Image.open(image_path)
    
    # Check if the image's height is greater than its width
    if img.height > img.width:
        # If true, rotate the image counterclockwise by 90 degrees
        img = img.rotate(90, expand=True)
        # Save and overwrite the original image
        img.save(image_path)


def rotate_video_if_tall(video_path):
    """
    If the video's width is less than its height, rotate it counterclockwise by 90 degrees.
    
    Parameters:
    video_path: Path to the video.
    """
    # Open the video using moviepy
    clip = VideoFileClip(video_path)
    
    # Check if the video's width is less than its height
    if clip.size[0] < clip.size[1]:
        # If true, rotate the video counterclockwise by 90 degrees
        rotated_clip = clip.rotate(90)
        video_path_tmp = video_path.split('.')[0] + '_tmp' + '.mp4'
        print("ROTA the video:",  video_path_tmp)
        # Save the modified video to the original path or a new path
        rotated_clip.write_videofile(video_path_tmp, codec="libx264", audio_codec="aac")

# def main_worker():
