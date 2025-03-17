import os
from moviepy.editor import ImageSequenceClip

# Set the image folder path and the output video file path
image_folder = '/path/to/image'  # Replace with your image folder path
output_video = '/path/to/video.mp4'  # Set the output video file name

# Get the list of image files and sort them by filename
images = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith('.png')]
images.sort()
print(len(images))

# Set the video frame rate (number of images displayed per second)
fps = 24  # You can adjust this value as needed

# Create the video clip
clip = ImageSequenceClip(images, fps=fps)

# Write the video file
clip.write_videofile(output_video, codec='libx264')