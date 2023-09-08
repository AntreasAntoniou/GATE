data_dir = "/data-fast0/"

from gate.data.medical.segmentation.automated_cardiac_diagnosis import (
    build_gate_dataset,
)

# from gate.data.medical.segmentation.medical_decathlon import build_gate_dataset

data = build_gate_dataset(
    data_dir=data_dir, image_size=64, target_image_size=64
)


from torch.utils.data import DataLoader

dataloader = DataLoader(
    data["train"], batch_size=1, shuffle=True, num_workers=12
)


from tqdm.auto import tqdm

for idx, batch in tqdm(enumerate(dataloader)):
    for key, value in batch.items():
        print(key, value.shape)
    if idx > 100:
        break
