from pathlib import Path

import rasterio
from matplotlib import pyplot as plt
from rich import print

# data = huggingface_hub.snapshot_download(
#     repo_id="GATE-engine/GeoLifeCLEF2023",
#     repo_type="dataset",
#     use_auth_token=True,
#     resume_download=True,
#     max_workers=mp.cpu_count(),
#     cache_dir=os.environ.get("DATASET_DIR", "/tmp"),
#     local_dir=os.environ.get("DATASET_DIR", "/tmp"),
# )


def visualize_tiff(file_path):
    # Open the raster file
    with rasterio.open(file_path) as src:
        # Read the raster data
        img = src.read(1)
    print(img.shape)
    # Display the raster data
    plt.imshow(img, cmap="viridis")
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    print(f"Visualizing {Path(__file__).name}")
    visualize_tiff(
        file_path=Path(
            "/disk/scratch_fast1/data/GeoLifeCLEF2023/climate/Climate/BioClimatic_Average_1981-2010/bio1.tif"
        )
    )
