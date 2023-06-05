import learn2learn as l2l
from tqdm.auto import tqdm
import torchvision.transforms as T
import datasets

dataset_root = "/data/"

dataset_dict = {
    # "aircraft_bbcrop": lambda set_name: l2l.vision.datasets.FGVCAircraft(
    #     root=dataset_root,
    #     mode=set_name,
    #     download=True,
    #     bounding_box_crop=True,
    # ),
    # "cubirds200_bbcrop": lambda set_name: l2l.vision.datasets.CUBirds200(
    #     root=dataset_root,
    #     mode=set_name,
    #     download=True,
    #     bounding_box_crop=True,
    # ),
    # "describable_textures": lambda set_name: l2l.vision.datasets.DescribableTextures(
    #     root=dataset_root,
    #     mode=set_name,
    #     download=True,
    # ),
    # "mini_imagenet": lambda set_name: l2l.vision.datasets.MiniImagenet(
    #     root=dataset_root,
    #     mode=set_name,
    #     download=True,
    # ),
    "omniglot": lambda set_name: l2l.vision.datasets.FullOmniglot(
        root=dataset_root,
        download=True,
        set_name=set_name,
    ),
    "vggflowers": lambda set_name: l2l.vision.datasets.VGGFlower102(
        root=dataset_root,
        mode=set_name,
        download=True,
    ),
    "fungi": lambda set_name: l2l.vision.datasets.FGVCFungi(
        root=dataset_root,
        mode=set_name,
        download=True,
    ),
}

if __name__ == "__main__":
    with tqdm(total=len(dataset_dict)) as pbar_dataset:
        for key, value in dataset_dict.items():
            hf_dataset_dict = dict()
            with tqdm(
                total=len(["train", "validation", "test"])
            ) as pbar_set_name:
                pbar_dataset.set_description(f"Processing {key}")
                for set_name in ["train", "validation", "test"]:
                    pbar_set_name.set_description(f"Processing {set_name}")
                    dataset = value(set_name=set_name)
                    data_dict = {"image": [], "label": []}
                    with tqdm(total=len(dataset)) as pbar_data:
                        for idx, item in enumerate(dataset):
                            data_dict["image"].append(
                                T.Resize(size=(224, 224))(item[0])
                            )
                            data_dict["label"].append(item[1])
                            pbar_data.update(1)

                    hf_dataset = datasets.Dataset.from_dict(data_dict)
                    hf_dataset_dict[set_name] = hf_dataset
                    pbar_set_name.update(1)
            hf_dataset_dict_full = datasets.DatasetDict(hf_dataset_dict)
            hf_dataset_dict_full.push_to_hub(
                repo_id=f"Antreas/{key}", private=False
            )
            pbar_dataset.update(1)
