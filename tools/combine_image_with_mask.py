from PIL import Image


def overlay_mask(image_path, mask_path, output_path):
    # 打开图片和mask
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")  # 将mask转换为灰度模式

    # 创建纯红色遮罩
    red_overlay = Image.new("RGBA", image.size, (0, 0, 255, 200))  # 初始为完全透明的红色

    # 将mask应用为遮罩的alpha通道
    red_overlay.putalpha(mask)  # 使用mask作为alpha通道，使得mask区域呈现红色

    # 将红色遮罩覆盖到原图上
    image_with_overlay = Image.alpha_composite(image.convert("RGBA"), red_overlay)

    # 保存结果
    image_with_overlay.convert("RGB").save(output_path, "PNG")


def combine_image_with_mask(image_path, mask_path, output_path):
    # 打开图片和mask
    image = Image.open(image_path)
    mask = Image.open(mask_path)

    # 获取mask的调色板
    palette = mask.getpalette()

    # 创建一个新的调色板
    new_palette = []

    # 遍历调色板，每四个元素代表一个颜色（红、绿、蓝）
    for i in range(0, len(palette), 4):
        # 假设我们想要将调色板中的某个特定颜色（例如红色）替换为半透明的红色
        # 这里我们需要知道具体是哪个颜色，这里假设是调色板中的第一个颜色（索引0）
        if i == 0:  # 替换第一个颜色
            new_palette.extend((255, 0, 0))  # 红色
            new_palette.append(128)  # 50%透明度
        else:
            new_palette.extend(palette[i:i+4])  # 复制原始颜色

    # 应用新的调色板
    mask.putpalette(new_palette)

    # 将mask转换为RGBA模式
    mask = mask.convert("RGBA")

    # 创建一个新的透明图片，尺寸与原图相同
    new_image = Image.new("RGBA", image.size)

    # 将原图和mask合成到新图片上
    new_image.paste(image, (0, 0), mask)

    # 保存合成后的图片
    new_image.save(output_path, "PNG")
    
if __name__ == "__main__":
    overlay_mask(image_path="/home/bingxing2/home/scx8ah2/dataset/ROVES_summary/ROVES_week_13/JPEGImages/0514_pour_tea_17/0000300.jpg",
                            mask_path="/home/bingxing2/home/scx8ah2/dataset/ROVES_summary/ROVES_week_13/Annotations/0514_pour_tea_17/0000300.png",
                            output_path="0514_pour_tea_17_0000300.png")