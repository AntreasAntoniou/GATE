import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt


def pad_image(image, target_size=224):
    w, h = image.size
    pad_w = (target_size - w) // 2
    pad_h = (target_size - h) // 2
    padding = (pad_w, pad_h, target_size - w - pad_w, target_size - h - pad_h)
    return transforms.functional.pad(image, padding, fill=0)


# Load your 32x32 image as a PIL image
image_path = "path/to/your/image.jpg"
image = Image.open(image_path)

# Apply the custom pad_image transform
padded_image = pad_image(image, target_size=224)

# Visualize the original and padded images
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
ax1.imshow(image)
ax1.set_title("Original Image")
ax1.axis("off")
ax2.imshow(padded_image)
ax2.set_title("Padded Image")
ax2.axis("off")
plt.show()
