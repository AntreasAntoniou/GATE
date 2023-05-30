import learn2learn as l2l
from tqdm.auto import tqdm

data = l2l.vision.datasets.FGVCAircraft(
    root="data/",
    mode="all",
    download=True,
    bounding_box_crop=False,
)

label_set = set()

with tqdm(total=len(data)) as pbar:
    for item in data:
        label_set.add(item[1])
        pbar.update(1)
        pbar.set_description(f"Found {len(label_set)} labels")
