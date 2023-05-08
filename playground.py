from imguruploader import ImgurUploader
import os
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt


def pad_image(image, target_size=224):
    w, h = image.size
    pad_w = (target_size - w) // 2
    pad_h = (target_size - h) // 2
    padding = (pad_w, pad_h, target_size - w - pad_w, target_size - h - pad_h)
    return transforms.functional.pad(image, padding, fill=0)


def create_sample_image(size=(32, 32)):
    image = Image.new("RGB", size, color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

    for i in range(0, size[0], 8):
        for j in range(0, size[1], 8):
            draw.rectangle(
                [i, j, i + 7, j + 7], fill=colors[((i // 8) + (j // 8)) % 3]
            )

    return image


def upload_to_imgur(image, client_id):
    uploader = ImgurUploader(client_id)
    image_path = "temp_image.png"
    image.save(image_path)
    response = uploader.upload_image(image_path)
    os.remove(image_path)
    return response["link"]


client_id = "4fa27e2baf27c1e"  # Replace with your actual Imgur Client ID

# Create a 32x32 sample image
image = create_sample_image()


# Apply the custom pad_image transform
padded_image = pad_image(image, target_size=224)

# Upload the original and padded images to Imgur
image_url = upload_to_imgur(image, client_id)
padded_image_url = upload_to_imgur(padded_image, client_id)

print("Original image URL:", image_url)
print("Padded image URL:", padded_image_url)
