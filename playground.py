import imgur_uploader
import os
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

os.environ["IMGUR_API_ID"] = "4fa27e2baf27c1e"
os.environ["IMGUR_API_SECRET"] = "4d8c4377cfc0fbda33c454b46d227621b04e078d"


def pad_image(image, target_size=224):
    w, h = image.size
    pad_w = (target_size - w) // 2
    pad_h = (target_size - h) // 2
    padding = (pad_w, pad_h, target_size - w - pad_w, target_size - h - pad_h)
    return transforms.functional.pad(image, padding, fill=0)


def create_sample_image(size=(32, 32)):
    image = Image.new("RGB", size, color=(128, 128, 255))

    return image


def upload_to_imgur(image):
    image_path = "temp_image.png"
    image.save(image_path)
    response = imgur_uploader.upload_image(image_path)
    os.remove(image_path)
    return response["link"]


if __name__ == "__main__":
    # Create a 32x32 sample image
    image = create_sample_image()

    # Apply the custom pad_image transform
    padded_image = pad_image(image, target_size=224)

    # Upload the original and padded images to Imgur
    image_url = upload_to_imgur(image)
    padded_image_url = upload_to_imgur(padded_image)

    print("Original image URL:", image_url)
    print("Padded image URL:", padded_image_url)
