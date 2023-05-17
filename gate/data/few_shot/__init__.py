def bytes_to_string(x):
    return (
        x.decode("utf-8").lower() if isinstance(x, bytes) else str(x).lower()
    )


from .aircraft import AircraftFewShotClassificationDataset
from .cifarfs import CIFARFewShotClassificationDataset
from .cubirds200 import CUB200FewShotClassificationDataset
from .describable_textures import (
    DescribableTexturesFewShotClassificationDataset,
)
from .fc100 import FC100FewShotClassificationDataset
from .fungi import FungiFewShotClassificationDataset
from .mini_imagenet import MiniImageNetFewShotClassificationDataset
from .omniglot import OmniglotFewShotClassificationDataset
from .quickdraw import QuickDrawFewShotClassificationDataset
from .tiered_imagenet import TieredImageNetFewShotClassificationDataset
from .vggflowers import VGGFlowersFewShotClassificationDataset
