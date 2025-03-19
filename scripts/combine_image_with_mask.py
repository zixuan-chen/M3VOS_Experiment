from PIL import Image


def overlay_mask(image_path, mask_path, output_path):
    # Open the image and mask
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")  # Convert the mask to grayscale mode

    # Create a pure red overlay
    red_overlay = Image.new("RGBA", image.size, (0, 0, 255, 200))  # Initially fully transparent red

    # Apply the mask as the alpha channel of the overlay
    red_overlay.putalpha(mask)  # Use the mask as the alpha channel to make the mask area appear red

    # Overlay the red mask onto the original image
    image_with_overlay = Image.alpha_composite(image.convert("RGBA"), red_overlay)

    # Save the result
    image_with_overlay.convert("RGB").save(output_path, "PNG")


def combine_image_with_mask(image_path, mask_path, output_path):
    # Open the image and mask
    image = Image.open(image_path)
    mask = Image.open(mask_path)

    # Get the mask's palette
    palette = mask.getpalette()

    # Create a new palette
    new_palette = []

    # Iterate through the palette, every four elements represent a color (red, green, blue)
    for i in range(0, len(palette), 4):
        # Assuming we want to replace a specific color in the palette (e.g., red) with a semi-transparent red
        # Here we need to know which specific color, assuming it's the first color in the palette (index 0)
        if i == 0:  # Replace the first color
            new_palette.extend((255, 0, 0))  # Red
            new_palette.append(128)  # 50% transparency
        else:
            new_palette.extend(palette[i:i+4])  


    mask.putpalette(new_palette)


    mask = mask.convert("RGBA")

    new_image = Image.new("RGBA", image.size)

    new_image.paste(image, (0, 0), mask)

    new_image.save(output_path, "PNG")
    
if __name__ == "__main__":
    overlay_mask(image_path="/path/to/ROVES/JPEGImages/0514_pour_tea_17/0000300.jpg",
                            mask_path="/path/to/ROVES/Annotations/0514_pour_tea_17/0000300.png",
                            output_path="0514_pour_tea_17_0000300.png")