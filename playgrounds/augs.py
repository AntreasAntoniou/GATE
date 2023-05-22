from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
from PIL import Image
from timm.data.auto_augment import rand_augment_transform
from torchvision.transforms import Compose, ToPILImage

# Download an image from the internet
url = "https://upload.wikimedia.org/wikipedia/commons/4/47/PNG_transparency_demonstration_1.png"
response = requests.get(url)
img = Image.open(BytesIO(response.content))

# Display the original image
plt.imshow(img)
plt.title("Original Image")
plt.show()

# Save the original image
img.save("original.png")

original_img = Image.open("original.png").convert("RGB")

# Define a transformation
hparams = {}
transform = Compose(
    [
        rand_augment_transform("rand-m9-mstd0.5-inc1", hparams={}),
    ]
)

all_images = [np.asarray(original_img)]
for _ in range(16):
    # Apply the transformation to image
    augmented_img = transform(original_img)

    # Convert back to PIL Image and append to list
    all_images.append(np.asarray(augmented_img))

# Convert list to numpy array
all_images = np.array(all_images)

# Compute number of images per row
n = int(np.sqrt(all_images.shape[0]))
image_dim = all_images.shape[-1]
# Tile images using numpy
tiled_image = np.block([[all_images[i * n : i * n + n]] for i in range(n)])

# Reshape tiled_image to 3D (for RGB images)
tiled_image = np.reshape(
    tiled_image, (n * all_images.shape[1], n * all_images.shape[2], image_dim)
)

# Convert back to PIL Image
tiled_image = Image.fromarray(tiled_image.astype("uint8"))

# Save the image
tiled_image.save("augmented_images.jpg")
