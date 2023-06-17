import requests
from PIL import Image
from transformers import SamModel, SamProcessor

model = SamModel.from_pretrained("facebook/sam-vit-base").to("cuda")
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

img_url = "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
input_points = []  # 2D localization of a window

inputs = processor([raw_image, raw_image], return_tensors="pt").to("cuda")

outputs = model(**inputs)
masks = processor.image_processor.post_process_masks(
    outputs.pred_masks.cpu(),
    inputs["original_sizes"].cpu(),
    inputs["reshaped_input_sizes"].cpu(),
)
print(outputs[0].shape)

# decoder_config = SamMaskDecoderConfig()
# decoder = SamMaskDecoder(config=self.decoder_config)
