from PIL import Image
import numpy as np
from skimage.feature import local_binary_pattern
from skimage.measure import shannon_entropy
from skimage import io
import matplotlib.pyplot as plt
import os
import re
import concurrent.futures
from tqdm import tqdm

def calculate_lbp_entropy(image):
    # 计算局部二值模式
    radius = 4
    n_points = 8 * radius
    lbp = local_binary_pattern(image, P=n_points, R=radius, method="uniform")
    # 计算LBP的直方图
    hist, _ = np.histogram(lbp.ravel(), bins=n_points+3, density=True)

    # 计算熵
    lbp_entropy = shannon_entropy(hist)

    return lbp_entropy, lbp, hist


def get_entropy_for_p_image(image_path):
    assert os.path.exists(image_path)
    image = Image.open(image_path).convert('L')
    image_array = np.array(image)
    pixels = np.unique(image_array)
    entropy_list = []
    # print("pixels: ", pixels)
    for i in pixels:
        if i:
            # print("i=", i)
            binary_image = (image_array == i).astype(np.uint8)
            entropy, lbp, lbp_hist = calculate_lbp_entropy(binary_image)
            # print("LBP Entropy (Measure of Chaos):", entropy)

            # # 可视化二值图、LBP 图和直方图
            # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

            # # 显示原始二值图
            # ax1.imshow(binary_image, cmap='gray')
            # ax1.set_title("Original Binary Image")
            # ax1.axis("off")

            # # 显示 LBP 图
            # ax2.imshow(lbp, cmap='gray')
            # ax2.set_title("LBP Image")
            # ax2.axis("off")

            # # 绘制 LBP 直方图
            # ax3.bar(range(len(lbp_hist)), lbp_hist, color='gray')
            # ax3.set_title("LBP Histogram")
            # ax3.set_xlabel("LBP Pattern")
            # ax3.set_ylabel("Frequency")

            # plt.tight_layout()
            # plt.savefig(f"lbp_for_{i}.png")
            # plt.close()
            entropy_list.append(entropy)
    # print("entropy list: ", entropy_list)
    if entropy_list:
        return np.mean(entropy_list)
    else:
        return np.nan



def get_entropy_for_dir(dir_path):
    file_num_to_entropy = {}
    pbar = tqdm(os.listdir(dir_path), desc=f"processing {dir_path.split('/')[-1]}")
    num = 0
    for filename in pbar:
        # print("calculating", filename)
        if filename.endswith(".png"):
            match = re.search(r"\d+", filename)
            if match:
                num = int(match.group())
                entropy = get_entropy_for_p_image(os.path.join(dir_path, filename))
                # 输出熵值
                if not np.isnan(entropy):
                    file_num_to_entropy[num] = entropy
            else:
                print("find unmatched files:", filename)
    
    nums = list(file_num_to_entropy.keys())
    if num > 1:

        midpoint = len(nums) // 2
        sorted_nums = sorted(nums)
        sorted_entropy = [file_num_to_entropy[n] for n in sorted_nums]

        first_half_avg = np.mean(sorted_entropy[:midpoint])
        second_half_avg = np.mean(sorted_entropy[midpoint:])
    
        # print(f"前50%随机数的平均值: {first_half_avg}")
        # print(f"后50%随机数的平均值: {second_half_avg}")

        return first_half_avg, second_half_avg
    else:
        print("Warning:folders founded have less than 2 images:", dir_path)
        return np.nan, np.nan

def get_entropy_for_folders(subfolder_paths):
    with concurrent.futures.ThreadPoolExecutor() as executor:
    # 使用map函数，它会自动处理线程的启动和返回值的收集
        results = list(executor.map(get_entropy_for_dir, subfolder_paths))
    
    first_half_avgs = [result[0] for result in results]
    second_half_avgs = [result[1] for result in results]

    return first_half_avgs, second_half_avgs

def get_entropy_for_dataset(dataset_name):
    assert dataset_name in ["roves", "vost", "d17", "y19"]
    if dataset_name == "roves":
        valid_weeks = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13]
        subfolder_paths = []
        for week in valid_weeks:
            print("collecting week", week)
            base_dir = f"/home/bingxing2/home/scx8ah2/dataset/ROVES_summary/ROVES_week_{week}/Annotations"
            
            subfolder_paths.extend([os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])

    elif dataset_name == "vost":
        base_dir = "/home/bingxing2/home/scx8ah2/dataset/VOST/Annotations"
        subfolder_paths = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    elif dataset_name == "d17":
        base_dir = "/home/bingxing2/home/scx8ah2/dataset/DAVIS/2017/Annotations/480p"
        subfolder_paths = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    elif dataset_name == "y19":
        base_dir = "/home/bingxing2/home/scx8ah2/dataset/YouTube/valid/Annotations"
        subfolder_paths = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]


    print("Start computing")
    first_avgs, second_avgs = get_entropy_for_folders(subfolder_paths)
    first_avgs_clean = [x for x in first_avgs if not np.isnan(x)]
    second_avgs_clean = [x for x in second_avgs if not np.isnan(x)]
    first_half_mean = np.mean(first_avgs_clean)
    second_half_mean = np.mean(second_avgs_clean)
    print("Finished")

    with open(f'{dataset_name}_entropy_result.txt', "w") as f:
        f.write(f"--------------------Result for {dataset_name}: ----------------------------")
        f.write(f"first half:{first_half_mean}, second half: {second_half_mean}")

    print(f"--------------------Result for {dataset_name}: ----------------------------")
    print(f"first half:{first_half_mean}, second half: {second_half_mean}")

def paint_entrogy_zhexian(dir_path):
    # 画出熵值随时间的变化，得到一个折线图
    file_num_to_entropy = {}
    pbar = tqdm(os.listdir(dir_path), desc=f"processing {dir_path.split('/')[-1]}")
    num = 0
    for filename in pbar:
        # print("calculating", filename)
        if filename.endswith(".png"):
            match = re.search(r"\d+", filename)
            if match:
                num = int(match.group())
                entropy = get_entropy_for_p_image(os.path.join(dir_path, filename))
                # 输出熵值
                if not np.isnan(entropy):
                    file_num_to_entropy[num] = entropy
            else:
                print("find unmatched files:", filename)

    nums = list(file_num_to_entropy.keys())

    if num > 1:
        sorted_nums = sorted(nums)
        sorted_entropy = [file_num_to_entropy[n] for n in sorted_nums]

        plt.plot(sorted_nums, sorted_entropy)
        plt.xlabel("frame number")
        plt.ylabel("entropy")
        plt.title(f"entropy change for {dir_path.split('/')[-1]}")
        # 保存图片
        plt.savefig(f"{dir_path.split('/')[-1]}_entropy.png")


if __name__ == "__main__":
    # print("start state")
    # get_entropy_for_p_image("/home/bingxing2/home/scx8ah2/dataset/ROVES_summary/ROVES_week_6/Annotations/coagulate_water_1/0000000.png")
    # image_path = "/home/bingxing2/home/scx8ah2/dataset/ROVES_summary/ROVES_week_3/Annotations/break_puzzle_1/0000099.png"
    # print(image_path)
    # get_entropy_for_p_image(image_path)
    # dir_path = "/home/bingxing2/home/scx8ah2/dataset/ROVES_summary/ROVES_week_6/Annotations"
    # first_halfs, second_halfs = get_entropy_for_dataset(dir_path)
    # first_half_mean = np.mean(first_halfs)
    # second_half_mean = np.mean(second_halfs)

    # print("--------------------Result for ROVES week 6: ----------------------------")
    # print(f"first half:{first_half_mean}, second half: {second_half_mean}")
    # get_entropy_for_dataset("roves")

    print("half entropy for DAVIS 2017")
    # print(get_entropy_for_dir("/home/bingxing2/home/scx8ah2/jiaxin/DeformVOS/methods/TAM-VT-main/datasets/DAVIS/2017/Annotations/480p/gold-fish"))
    # paint_entrogy_zhexian("/home/bingxing2/home/scx8ah2/jiaxin/DeformVOS/methods/TAM-VT-main/datasets/DAVIS/2017/Annotations/480p/gold-fish")


    print("11.png")
    print(get_entropy_for_p_image("/home/bingxing2/home/scx8ah2/jiaxin/DeformVOS/methods/TAM-VT-main/datasets/DAVIS/2017/Annotations/480p/gold-fish/00011.png"))

    print("45.png")
    print(get_entropy_for_p_image("/home/bingxing2/home/scx8ah2/jiaxin/DeformVOS/methods/TAM-VT-main/datasets/DAVIS/2017/Annotations/480p/gold-fish/00045.png"))
j
    print("77.png")
    print(get_entropy_for_p_image("/home/bingxing2/home/scx8ah2/jiaxin/DeformVOS/methods/TAM-VT-main/datasets/DAVIS/2017/Annotations/480p/gold-fish/00077.png"))
    