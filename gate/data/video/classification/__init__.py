# Update the DatasetNames Enum to assign specific string names to each dataset
from enum import Enum


class DatasetNames(Enum):
    HMDB51_GULPRGB = "hmdb51-gulprgb"
    UCF_101_GULPRGB = "ucf-101-gulprgb"
    EPIC_KITCHENS_100_GULPRGB = "epic-kitchens-100-gulprgb"
    KINETICS_400 = "kinetics-400"
