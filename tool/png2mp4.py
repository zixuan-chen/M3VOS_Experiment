import os
from moviepy.editor import ImageSequenceClip

# 设置图片文件夹路径和输出视频文件路径
image_folder = '/home/lijiaxin/Deform_VOS/DeformVOS/tmp/7_squeeze_pasta'  # 替换为你的图片文件夹路径
output_video = '/home/lijiaxin/Deform_VOS/DeformVOS/tmp/test.mp4'  # 设置输出视频文件名

# 获取图片文件列表，并按文件名排序
images = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith('.png')]
images.sort()
print(len(images))
# 设置视频帧率（每秒显示的图片数量）
fps = 24  # 你可以根据需要调整这个值

# 创建视频剪辑
clip = ImageSequenceClip(images, fps=fps)

# 写入视频文件
clip.write_videofile(output_video, codec='libx264')
