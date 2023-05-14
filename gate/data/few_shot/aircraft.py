from typing import Optional
import learn2learn as l2l


def build_aircraft_dataset(
    set_name: str,
    data_dir: Optional[str] = None,
    bounding_box_crop: bool = True,
    num_tasks: int = 1000000,
):
    dataset = l2l.vision.datasets.FGVCAircraft(
        root=data_dir,
        mode=set_name,
        download=True,
        bounding_box_crop=bounding_box_crop,
    )

    dataset = l2l.data.MetaDataset(dataset)
    dataset = l2l.data.TaskDataset(dataset=dataset, num_tasks=num_tasks)
