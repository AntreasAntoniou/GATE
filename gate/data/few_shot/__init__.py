from enum import Enum


def bytes_to_string(x):
    return (
        x.decode("utf-8").lower() if isinstance(x, bytes) else str(x).lower()
    )


class DatasetName(Enum):
    # AIRFS = "aircraft-fs-classification"
    CUBFS = "cubirds-fs-classification"
    # DTEXTFS = "dtextures-fs-classification"
    # FUNGIFS = "fungi-fs-classification"
    # MINIINFS = "mini-imagenet-fs-classification"
    # OMNIFS = "omniglot-fs-classification"
    # VGGFS = "vgg-flowers-fs-classification"
