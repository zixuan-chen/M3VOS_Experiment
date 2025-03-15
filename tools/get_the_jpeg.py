import cv2
import os

def extract_frames(video_path, output_folder):
    # 创建输出文件夹（如果不存在）
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 保存每一帧
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

    cap.release()
    print(f"Extracted {frame_count} frames to {output_folder}")

# 示例用法
if __name__ == "__main__":
    import os 
    # model_ls = ["SAM2", "Cutie", "Ours", "GT", "Rmem"]
    model_ls = ["GT"]
    video_id = [ "assemble_puzzle_3.mp4"]
    # model_ls = ["SAM2"]
    video_id = [ "0538"]
    data_root = r"./tmp/draw_faliure_case_in_supple_2"


    for model in model_ls:
        model_path = os.path.join(data_root, model)
        for video_name in os.listdir(model_path):
            if video_name.split('_')[0].isdigit():
                if video_name.split('_')[0] not in video_id:
                    continue
            else:
                # print("DONE", video_name )
                if video_name not in video_id:
                    continue     

            output_folder = os.path.join(data_root, "failure_case_jpeg", model)
            os.makedirs(output_folder, exist_ok=True)
            extract_frames(os.path.join(model_path,video_name), output_folder=os.path.join(output_folder, video_name))

            print("Output in " , os.path.join(output_folder, video_name))
    
    
    # video_path = 'path/to/your/video.mp4'
    # output_folder = 'path/to/output/folder'
    # extract_frames(video_path, output_folder)